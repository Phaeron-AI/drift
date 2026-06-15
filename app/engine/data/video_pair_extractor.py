from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import HybridConfig
from ..models.backbones import FrozenSam2

logger = logging.getLogger("drift.video_pairs")


@dataclass
class PairConfig:
  frame_gap: int = 8
  min_disp_px: float = 8.0
  max_disp_frac: float = 0.5
  min_mask_area_frac: float = 0.01
  max_mask_area_frac: float = 0.6


def _mask_centroid(mask: Tensor) -> Optional[tuple[float, float]]:
  if mask.ndim == 3:
    mask = mask.squeeze(0)
  ys, xs = mask.nonzero(as_tuple=True)
  if len(xs) == 0:
    return None
  return (xs.float().mean().item(), ys.float().mean().item())


def _pick_object_prompt(first_frame: Tensor) -> Tensor:
  h, w = first_frame.shape[-2], first_frame.shape[-1]
  return torch.tensor([[w / 2.0, h / 2.0]], device=first_frame.device, dtype=torch.float32)


def _iter_frames(video_path: str, cfg: HybridConfig, frame_gap: int):
  try:
    import decord
  except ImportError:
    raise ImportError("Please install decord: pip install decord")

  decord.bridge.set_bridge("torch")
  vr = decord.VideoReader(video_path, ctx=decord.cpu())
  total = len(vr)
  size = cfg.img_size

  for frame_idx in range(0, total - frame_gap, frame_gap):
    frames = vr.get_batch([frame_idx, frame_idx + frame_gap])  # [2, H, W, 3] uint8
    fa = frames[0].permute(2, 0, 1).float() / 255.0            # [3, H, W] native res
    fb = frames[1].permute(2, 0, 1).float() / 255.0
    # resize to img_size (bilinear); add/remove batch dim for F.interpolate
    fa = F.interpolate(fa.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    fb = F.interpolate(fb.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    yield frame_idx, fa.to(cfg.device), fb.to(cfg.device)


def extract_pairs(video_path: str, sam2: FrozenSam2, cfg: HybridConfig,
                  pair_cfg: PairConfig) -> Iterator[dict]:
  stem = Path(video_path).stem
  for frame_idx, frame_a, frame_b in _iter_frames(video_path, cfg, pair_cfg.frame_gap):
    prompt_a = _pick_object_prompt(frame_a)
    mask_a = sam2.predict_mask(frame_a.unsqueeze(0), prompt_a)[0, 0]   # [H, W] at img_size
    ca = _mask_centroid(mask_a)
    if ca is None:
      continue

    # track the SAME object into B by prompting at A's centroid (simple-start; upgrade to
    # SAM2 video propagation if fast objects outrun this).
    prompt_b = torch.tensor([[ca[0], ca[1]]], device=frame_b.device, dtype=torch.float32)
    mask_b = sam2.predict_mask(frame_b.unsqueeze(0), prompt_b)[0, 0]
    cb = _mask_centroid(mask_b)
    if cb is None:
      continue

    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    disp = (dx * dx + dy * dy) ** 0.5
    H = frame_a.shape[-2]                      # == cfg.img_size now
    area_a = mask_a.float().mean().item()

    if disp < pair_cfg.min_disp_px:            # no motion -> no signal
      continue
    if disp > pair_cfg.max_disp_frac * H:      # too much -> cut / object left frame
      continue
    if not (pair_cfg.min_mask_area_frac <= area_a <= pair_cfg.max_mask_area_frac):
      continue                                 # too small (unreliable) / whole-frame (camera motion)

    yield {
      "id": f"{stem}_{frame_idx:06d}",         # stable key for the cache (no path on video samples)
      "source_image": frame_a.cpu(),           # [3, img_size, img_size] in [0,1]
      "target_image": frame_b.cpu(),
      "start_x": ca[0], "start_y": ca[1],      # coords in img_size space (matches precompute)
      "end_x": cb[0], "end_y": cb[1],
    }


def build_video_dataset(video_paths: list[str], cfg: HybridConfig, pair_cfg: Optional[PairConfig] = None) -> Iterator[dict]:
  import dataclasses
  pair_cfg = pair_cfg or PairConfig()
  sam2 = FrozenSam2(dataclasses.replace(cfg, sam2_residency="resident"))
  kept = 0
  for vp in video_paths:
    for triple in extract_pairs(vp, sam2, cfg, pair_cfg):
      kept += 1
      yield triple
  logger.info(f"video pairs: yielded {kept} triples from {len(video_paths)} videos")