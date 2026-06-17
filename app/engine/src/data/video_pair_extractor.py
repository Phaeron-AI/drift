from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import HybridConfig
from ..models.backbones import FrozenSam2

logger = logging.getLogger("drift.video_pairs")


@dataclass
class PairConfig:              
  window: int = 16              
  stride: int = 16             
  cut_threshold: float = 0.2
  min_disp_px: float = 8.0      
  max_disp_frac: float = 0.5     
  min_mask_area_frac: float = 0.01   
  max_mask_area_frac: float = 0.6    
  max_area_ratio: float = 2.0     


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
  ys, xs = np.nonzero(mask)
  if len(xs) == 0:
    return None
  return (float(xs.mean()), float(ys.mean()))


def _pick_object_prompt(frame_a: Tensor, frame_b: Tensor) -> tuple[float, float]:
  diff = (frame_b - frame_a).abs().mean(dim=0)        # [H, W]
  idx = int(diff.flatten().argmax().item())
  W = diff.shape[1]
  return (float(idx % W), float(idx // W))


def _iter_windows(video_path: str, cfg: HybridConfig, pcfg: PairConfig):
  """Yield (start_idx, frames_uint8) where frames_uint8 is [window, size, size, 3] uint8 at img_size."""
  try:
    import decord
  except ImportError:
    raise ImportError("Please install decord: pip install decord")
  decord.bridge.set_bridge("torch")
  vr = decord.VideoReader(video_path, ctx=decord.cpu())
  total = len(vr)
  size = cfg.img_size

  for start in range(0, total - pcfg.window, pcfg.stride):
    idxs = list(range(start, start + pcfg.window))
    frames = vr.get_batch(idxs)                          # [window, H, W, 3] uint8
    f = frames.permute(0, 3, 1, 2).float()               # [window, 3, H, W]
    f = F.interpolate(f, size=(size, size), mode="bilinear", align_corners=False)
    f = f.clamp(0, 255).byte().permute(0, 2, 3, 1)       # [window, size, size, 3] uint8
    yield start, f


def _has_cut(frames_uint8: Tensor, threshold: float) -> bool:
  f = frames_uint8.float() / 255.0                       # [window, H, W, 3]
  diffs = (f[1:] - f[:-1]).abs().mean(dim=(1, 2, 3))     # [window-1]
  return bool((diffs > threshold).any().item())


def _write_frames_tmp(frames_uint8: Tensor, tmp_dir: Path) -> None:
  from PIL import Image
  for i, fr in enumerate(frames_uint8):
    Image.fromarray(fr.cpu().numpy()).save(tmp_dir / f"{i:05d}.jpg")


def extract_pairs(video_path: str, sam2: FrozenSam2, cfg: HybridConfig,
                  pair_cfg: Optional[PairConfig] = None) -> Iterator[dict]:
  pcfg = pair_cfg or PairConfig()
  stem = Path(video_path).stem
  last = pcfg.window - 1

  for start, frames in _iter_windows(video_path, cfg, pcfg):
    # 1) reject windows containing a scene cut (the image-5 failure)
    if _has_cut(frames, pcfg.cut_threshold):
      continue

    # 2) propagate ONE object across the cut-free window via SAM2 video mode
    fa = frames[0].permute(2, 0, 1).float() / 255.0
    fb = frames[1].permute(2, 0, 1).float() / 255.0
    px, py = _pick_object_prompt(fa, fb)
    with tempfile.TemporaryDirectory() as td:
      tmp = Path(td)
      _write_frames_tmp(frames, tmp)
      masks = sam2.propagate_object(tmp, (px, py), ann_frame_idx=0)

    # 3) require the object present in the first AND last frame
    m0, mL = masks.get(0), masks.get(last)
    if m0 is None or mL is None:
      continue
    c0, cL = _mask_centroid(m0), _mask_centroid(mL)
    if c0 is None or cL is None:
      continue

    # 4) area filters: sane size + stable (didn't jump to another object)
    a0 = float(m0.mean())
    aL = float(mL.mean())
    if not (pcfg.min_mask_area_frac <= a0 <= pcfg.max_mask_area_frac):
      continue
    ratio = max(a0, aL) / max(min(a0, aL), 1e-6)
    if ratio > pcfg.max_area_ratio:
      continue

    # 5) displacement filters
    dx, dy = cL[0] - c0[0], cL[1] - c0[1]
    disp = (dx * dx + dy * dy) ** 0.5
    if disp < pcfg.min_disp_px:                # no motion
      continue
    if disp > pcfg.max_disp_frac * cfg.img_size:   # too much (residual cut / track jump)
      continue

    # 6) emit -- SAME dict shape as before, so precompute is unchanged
    yield {
      "id": f"{stem}_{start:06d}",
      "source_image": frames[0].permute(2, 0, 1).float().cpu() / 255.0,   # [3,size,size] in [0,1]
      "target_image": frames[last].permute(2, 0, 1).float().cpu() / 255.0,
      "start_x": c0[0], "start_y": c0[1],
      "end_x": cL[0], "end_y": cL[1],
    }


def build_video_dataset(video_paths: list[str], cfg: HybridConfig,
                        pair_cfg: Optional[PairConfig] = None) -> Iterator[dict]:
  import dataclasses
  pcfg = pair_cfg or PairConfig()
  sam2 = FrozenSam2(dataclasses.replace(cfg, sam2_residency="resident"))
  kept = 0
  for vp in video_paths:
    for triple in extract_pairs(vp, sam2, cfg, pcfg):
      kept += 1
      yield triple
  logger.info(f"video pairs: yielded {kept} triples from {len(video_paths)} videos")