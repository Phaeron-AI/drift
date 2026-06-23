from __future__ import annotations

# =============================================================================
# MOVi-C -> triple adapter. The cleanest data source in the project: the trajectory and masks
# are GROUND TRUTH (no SAM, no centroid estimation). MOVi-C = realistic textured GSO objects on
# HDRI backgrounds, FIXED camera (object moves, camera does not -> trajectory is pure object
# motion, unlike the DAVIS camera-motion contamination). Its "test" split is purpose-built for
# generalization (held-out objects AND backgrounds) -> use it as the honest eval set.
#
# Key fields (per sample, 24 frames @ 256x256):
#   video[s,256,256,3] uint8         -> source/target frames (resized to img_size)
#   segmentations[s,256,256,1] uint8 -> per-pixel instance id, background=0.
#       NOTE: instance id = index+1 (id k corresponds to instances[k-1]).
#   instances/image_positions[k,s,2] -> NORMALIZED (0,1) center-of-mass per frame -> trajectory
#       is just image_positions[obj, j] - image_positions[obj, i].
#       AXIS: empirically verified against the segmentation-mask centroid (smoke test, 87/90
#       off-diagonal samples) that image_positions is (x, y) = (col, row) -- NOT (row, col) as
#       some Kubric docs state. So we unpack `xi, yi = img_pos[...]` directly. The 3 "disagree"
#       samples were near-diagonal objects where x~y and the check can't disambiguate. Lesson
#       (same as pathway_health): trust the cross-check against a known result, not the doc.
#   instances/visibility[k,s]        -> pixels visible per frame; require object visible in both.
#
# Drops into the existing pipeline: yields the SAME triple dict as davis_adapter (id, split,
# source/target_image, start/end x/y, source/target_mask), so precompute/train are unchanged.
# =============================================================================

import logging
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import HybridConfig

logger = logging.getLogger("drift.movi")


@dataclass
class MoviConfig:
  variant: str = "movi_c/256x256"
  data_dir: str = "gs://kubric-public/tfds"
  frame_gap: int = 6              # 24 fps-ish, 24 frames total; gap 6 = decent motion
  pairs_per_video: int = 3        # sample a few (i, i+gap) pairs per video
  min_disp_norm: float = 0.04     # min normalized object displacement to keep (skip near-static)
  max_videos: Optional[int] = None
  min_visible_px: int = 200       # object must occupy >= this many px in BOTH frames
  split: str = "train"            # "train" | "test" (test = held-out objects+backgrounds)
  seed: int = 0                   # local RNG seed -> reproducible pair selection across runs


def _resize_frame(arr_uint8: np.ndarray, size: int) -> torch.Tensor:
  # arr [256,256,3] uint8 -> [3,size,size] float [0,1]
  t = torch.from_numpy(arr_uint8).permute(2, 0, 1).float() / 255.0
  t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
  return t.squeeze(0)


def _resize_mask(binmask_uint8: np.ndarray, size: int) -> np.ndarray:
  # [256,256] {0,255} -> [size,size] {0,255} nearest (keep binary)
  from PIL import Image
  return np.array(Image.fromarray(binmask_uint8).resize((size, size), Image.Resampling.NEAREST))


def _pick_object(seg_i: np.ndarray, vis_i, vis_j, min_vis: int) -> Optional[int]:
  # choose the instance (id>=1) that is well-visible in BOTH frames and largest at source.
  ids, counts = np.unique(seg_i, return_counts=True)
  cand = [(int(i), int(c)) for i, c in zip(ids, counts) if i != 0 and c >= min_vis]
  if not cand:
    return None
  cand.sort(key=lambda ic: -ic[1])
  for inst_id, _ in cand:
    k = inst_id - 1   # instance id is index+1
    if k < len(vis_i) and vis_i[k] >= min_vis and vis_j[k] >= min_vis:
      return inst_id
  return None


def build_movi_dataset(cfg: HybridConfig, mcfg: Optional[MoviConfig] = None) -> Iterator[dict]:
  import tensorflow_datasets as tfds
  mcfg = mcfg or MoviConfig()
  size = cfg.img_size

  logger.info(f"MOVi: loading {mcfg.variant} split={mcfg.split} from {mcfg.data_dir}")
  ds = tfds.load(mcfg.variant, data_dir=mcfg.data_dir, split=mcfg.split)
  if mcfg.max_videos:
    ds = ds.take(mcfg.max_videos) # type: ignore

  rng = random.Random(mcfg.seed)  # local, seeded -> deterministic pair selection (not global RNG)
  n_videos, n_pairs = 0, 0
  for ex in tfds.as_numpy(ds):
    n_videos += 1
    video = ex["video"]                                   # [24,256,256,3] uint8
    seg = ex["segmentations"][..., 0]                     # [24,256,256] uint8 (drop channel)
    img_pos = ex["instances"]["image_positions"]          # [k,24,2] normalized (x,col),(y,row)
    vis = ex["instances"]["visibility"]                   # [k,24]
    n_frames = video.shape[0]
    vid_name = int(ex["metadata"]["video_name"])

    starts = list(range(0, n_frames - mcfg.frame_gap))
    rng.shuffle(starts)
    made = 0
    for i in starts:
      if made >= mcfg.pairs_per_video:
        break
      j = i + mcfg.frame_gap
      inst_id = _pick_object(seg[i], vis[:, i], vis[:, j], mcfg.min_visible_px)
      if inst_id is None:
        continue
      k = inst_id - 1
      # AXIS: image_positions is (x, y) = (col, row), verified against mask centroid. Unpack
      # x first, y second -- do NOT flip to (y, x) on the strength of the doc alone.
      xi, yi = img_pos[k, i]
      xj, yj = img_pos[k, j]
      # REJECT off-frame centers: MOVi objects get tossed and can leave the view, where
      # image_positions extrapolates outside [0,1] -> nonsense "teleport" trajectories. Require
      # both centers comfortably inside the frame (small margin) so the grab point and target
      # are real on-screen positions. (Caught by smoke test: a (565,150) grab = norm 1.1, off-frame.)
      m = 0.02
      if not (m <= xi <= 1 - m and m <= yi <= 1 - m and m <= xj <= 1 - m and m <= yj <= 1 - m):
        continue
      dx, dy = float(xj - xi), float(yj - yi)
      disp = (dx * dx + dy * dy) ** 0.5
      if disp < mcfg.min_disp_norm:
        continue

      sx, sy = float(xi) * size, float(yi) * size         # img_size-space grab point (x, y)
      ex_, ey_ = float(xj) * size, float(yj) * size
      src_mask = ((seg[i] == inst_id).astype(np.uint8) * 255)
      tgt_mask = ((seg[j] == inst_id).astype(np.uint8) * 255)

      yield {
        "id": f"movic_{mcfg.split}_{vid_name:06d}_{i:02d}_{inst_id}",
        "split": "val" if mcfg.split == "test" else "train",
        "source_image": _resize_frame(video[i], size),
        "target_image": _resize_frame(video[j], size),
        "start_x": sx, "start_y": sy, "end_x": ex_, "end_y": ey_,
        "source_mask": _resize_mask(src_mask, size),
        "target_mask": _resize_mask(tgt_mask, size),
      }
      made += 1
      n_pairs += 1

  logger.info(f"MOVi: {n_pairs} triples from {n_videos} videos "
              f"(variant={mcfg.variant}, split={mcfg.split}, gap={mcfg.frame_gap})")