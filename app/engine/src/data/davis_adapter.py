from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import HybridConfig

logger = logging.getLogger("drift.davis")


CURATED_PRISTINE = (
  # original pristine 12
  "bear", "blackswan", "boat", "camel", "car-shadow", "cows",
  "dog", "elephant", "flamingo", "goat", "rhino", "hike",
  # added: more single-salient-object clips with real translation
  "car-turn", "drift-straight", "drift-turn", "kite-walk", "lucia",
  "mallard-fly", "mallard-water", "rollerblade", "swing", "tennis",
  "dog-agility", "horsejump-low", "scooter-gray", "soapbox", "train",
  "bus", "motocross-bumps", "paragliding", "kite-surf", "lab-coat",
)


@dataclass
class DavisConfig:
  frame_gap: int = 6            # source->target gap (lower than before -> more pairs/clip)
  stride: int = 3             # step between emitted pairs (overlap -> many more pairs)
  min_disp_px: float = 6.0        # reject near-static pairs (img_size space)
  max_disp_frac: float = 0.6      # reject implausible jumps (occlusion swap / annotation gap)
  clips: Optional[tuple[str, ...]] = None   # None -> CURATED_PRISTINE


def _davis_roots(cfg: HybridConfig, davis_dir: Path) -> tuple[Path, Path]:
  jpeg = davis_dir / "JPEGImages" / "480p"
  anno = davis_dir / "Annotations" / "480p"
  if not jpeg.exists():
    raise FileNotFoundError(f"DAVIS frames not found at {jpeg}")
  if not anno.exists():
    raise FileNotFoundError(f"DAVIS annotations not found at {anno}")
  return jpeg, anno


def _read_mask_indices(path: Path) -> np.ndarray:
  return np.array(Image.open(path))


def _pick_instance(mask0: np.ndarray) -> Optional[int]:
  vals, counts = np.unique(mask0, return_counts=True)
  obj = [(v, c) for v, c in zip(vals.tolist(), counts.tolist()) if v != 0]
  if not obj:
    return None
  return max(obj, key=lambda vc: vc[1])[0]


def _centroid_scaled(mask: np.ndarray, inst: int, size: int) -> Optional[tuple[float, float]]:
  ys, xs = np.nonzero(mask == inst)
  if len(xs) == 0:
    return None
  h, w = mask.shape
  cx = float(xs.mean()) * size / w
  cy = float(ys.mean()) * size / h
  return (cx, cy)


def _load_frame(path: Path, size: int) -> torch.Tensor:
  img = Image.open(path).convert("RGB")
  t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
  t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
  return t.squeeze(0)


def extract_pairs(clip: str, jpeg_root: Path, anno_root: Path, cfg: HybridConfig,
                  dcfg: DavisConfig) -> Iterator[dict]:
  frames = sorted((jpeg_root / clip).glob("*.jpg"))
  masks = sorted((anno_root / clip).glob("*.png"))
  if len(frames) != len(masks) or len(frames) == 0:
    logger.warning(f"{clip}: frame/mask count mismatch ({len(frames)}/{len(masks)}); skipping")
    return
  size = cfg.img_size
  n = len(frames)

  for i in range(0, n - dcfg.frame_gap, dcfg.stride):
    j = i + dcfg.frame_gap
    m0 = _read_mask_indices(masks[i])
    mj = _read_mask_indices(masks[j])

    inst = _pick_instance(m0)
    if inst is None:
      continue
    c0 = _centroid_scaled(m0, inst, size)
    cj = _centroid_scaled(mj, inst, size)     # SAME instance index -> same object
    if c0 is None or cj is None:
      continue

    dx, dy = cj[0] - c0[0], cj[1] - c0[1]
    disp = (dx * dx + dy * dy) ** 0.5
    if disp < dcfg.min_disp_px:
      continue
    if disp > dcfg.max_disp_frac * size:
      continue

    yield {
      "id": f"{clip}_{i:05d}",
      "source_image": _load_frame(frames[i], size),
      "target_image": _load_frame(frames[j], size),
      "start_x": c0[0], "start_y": c0[1],
      "end_x": cj[0], "end_y": cj[1],
    }


def build_davis_dataset(cfg: HybridConfig, davis_dir: Path,
                        dcfg: Optional[DavisConfig] = None) -> Iterator[dict]:
  dcfg = dcfg or DavisConfig()
  jpeg_root, anno_root = _davis_roots(cfg, davis_dir)

  available = {d.name for d in jpeg_root.iterdir() if d.is_dir()}
  wanted = dcfg.clips if dcfg.clips is not None else CURATED_PRISTINE
  clips = [c for c in wanted if c in available]
  missing = [c for c in wanted if c not in available]
  if missing:
    logger.warning(f"requested clips not in dataset (skipped): {missing}")
  logger.info(f"DAVIS: {len(clips)} clips")

  kept = 0
  for clip in clips:
    for triple in extract_pairs(clip, jpeg_root, anno_root, cfg, dcfg):
      kept += 1
      yield triple
  logger.info(f"DAVIS: yielded {kept} triples from {len(clips)} clips "
              f"(gap={dcfg.frame_gap}, stride={dcfg.stride})")