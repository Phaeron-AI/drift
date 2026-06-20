from __future__ import annotations

import logging
import random
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
  "bear", "blackswan", "boat", "camel", "car-shadow", "cows",
  "dog", "elephant", "flamingo", "goat", "rhino", "hike",
  "car-turn", "drift-straight", "drift-turn", "kite-walk", "lucia",
  "mallard-fly", "mallard-water", "rollerblade", "swing", "tennis",
  "dog-agility", "horsejump-low", "scooter-gray", "soapbox", "train",
  "bus", "motocross-bumps", "paragliding", "kite-surf", "lab-coat",
)


@dataclass
class DavisConfig:
  frame_gap: int = 12           # dense full-DAVIS scaling run: moderate motion...
  stride: int = 5              # ...with dense overlap for VOLUME across all clips
  min_disp_px: float = 15.0       # keep real motion so the union mask stays active
  max_disp_frac: float = 0.8
  clips: Optional[tuple[str, ...]] = None   # None + use_all_clips=False -> CURATED_PRISTINE
  use_all_clips: bool = False     # True -> use EVERY clip in the dataset (full-DAVIS scaling)
  holdout_frac: float = 0.0       # fraction of CLIPS reserved as val (clip-level split)
  holdout_seed: int = 0           # reproducible split


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


def _instance_mask_resized(mask: np.ndarray, inst: int, size: int) -> np.ndarray:
  binm = (mask == inst).astype(np.uint8) * 255
  pil = Image.fromarray(binm).resize((size, size), Image.Resampling.NEAREST)
  return np.array(pil)


def _load_frame(path: Path, size: int) -> torch.Tensor:
  img = Image.open(path).convert("RGB")
  t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
  t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
  return t.squeeze(0)


def extract_pairs(clip: str, jpeg_root: Path, anno_root: Path, cfg: HybridConfig,
                  dcfg: DavisConfig, split: str) -> Iterator[dict]:
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
    cj = _centroid_scaled(mj, inst, size)
    if c0 is None or cj is None:
      continue
    dx, dy = cj[0] - c0[0], cj[1] - c0[1]
    disp = (dx * dx + dy * dy) ** 0.5
    if disp < dcfg.min_disp_px or disp > dcfg.max_disp_frac * size:
      continue

    yield {
      "id": f"{clip}_{i:05d}",
      "split": split,                                          # train / val (clip-level)
      "source_image": _load_frame(frames[i], size),
      "target_image": _load_frame(frames[j], size),
      "start_x": c0[0], "start_y": c0[1],
      "end_x": cj[0], "end_y": cj[1],
      "source_mask": _instance_mask_resized(m0, inst, size),
      "target_mask": _instance_mask_resized(mj, inst, size),
    }


def build_davis_dataset(cfg: HybridConfig, davis_dir: Path,
                        dcfg: Optional[DavisConfig] = None) -> Iterator[dict]:
  dcfg = dcfg or DavisConfig()
  jpeg_root, anno_root = _davis_roots(cfg, davis_dir)
  available = sorted([d.name for d in jpeg_root.iterdir() if d.is_dir()])

  if dcfg.use_all_clips:
    wanted = available
  else:
    wanted = [c for c in (dcfg.clips if dcfg.clips is not None else CURATED_PRISTINE)]
  clips = [c for c in wanted if c in available]
  missing = [c for c in wanted if c not in available]
  if missing:
    logger.warning(f"requested clips not in dataset (skipped): {missing}")

  # CLIP-LEVEL held-out split: reserve holdout_frac of clips ENTIRELY as val. Clip-level (not
  # frame-level) because frames from one clip are highly correlated -> a frame-level split would
  # leak. Generalization is only honestly tested on UNSEEN clips.
  val_clips: set[str] = set()
  if dcfg.holdout_frac > 0:
    rng = random.Random(dcfg.holdout_seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(dcfg.holdout_frac * len(shuffled))))
    val_clips = set(shuffled[:n_val])
    logger.info(f"HELD-OUT val clips ({len(val_clips)}): {sorted(val_clips)}")

  logger.info(f"DAVIS: {len(clips)} clips ({len(clips)-len(val_clips)} train / {len(val_clips)} val)")

  kept_tr, kept_val = 0, 0
  for clip in clips:
    split = "val" if clip in val_clips else "train"
    for triple in extract_pairs(clip, jpeg_root, anno_root, cfg, dcfg, split):
      if split == "val": kept_val += 1
      else: kept_tr += 1
      yield triple
  logger.info(f"DAVIS: yielded {kept_tr} train + {kept_val} val triples "
              f"(gap={dcfg.frame_gap}, stride={dcfg.stride})")