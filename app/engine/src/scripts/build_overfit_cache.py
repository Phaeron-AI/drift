from __future__ import annotations

# =============================================================================
# OVERFIT DIAGNOSTIC cache builder.
# Goal: test whether the architecture CAN learn trajectory->position AT ALL,
# stripped of every confound (data scale, generalization, variance).
#
# Construction that makes it a VALID test:
#   - ONE clip, FIXED source = frame 0.
#   - N targets = progressively later frames -> object travels further -> trajectory
#     GROWS and target POSITION shifts monotonically.
#   - Same source + different trajectory + different target position => the ONLY way to
#     fit all N is to USE the trajectory to place the object. The model cannot cheat by
#     memorizing appearance (source is identical) -- trajectory is forced to matter.
#
# Run from app/ :   python -m engine.src.scripts.build_overfit_cache --cache-dir overfit_cache
# Then train HARD on it:  ... train --cache-dir overfit_cache --max-steps 4000 --no-resume
# =============================================================================

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import HybridConfig
from ..models.backbones import DiffusionBackbone
from ..data.davis_adapter import (
  _davis_roots, _read_mask_indices, _pick_instance, _centroid_scaled,
  _instance_mask_resized, _load_frame, CURATED_PRISTINE,
)

logger = logging.getLogger("drift.overfit")
ENGINE_ROOT = Path(__file__).resolve().parent.parent


def _clip_travel(clip, jpeg_root, anno_root, size):
  """Total centroid displacement of frame0's instance from frame 0 -> last available frame."""
  frames = sorted((jpeg_root / clip).glob("*.jpg"))
  masks = sorted((anno_root / clip).glob("*.png"))
  if len(frames) != len(masks) or len(frames) < 8:
    return None
  m0 = _read_mask_indices(masks[0])
  inst = _pick_instance(m0)
  if inst is None:
    return None
  c0 = _centroid_scaled(m0, inst, size)
  cL = _centroid_scaled(_read_mask_indices(masks[-1]), inst, size)
  if c0 is None or cL is None:
    return None
  disp = ((cL[0] - c0[0]) ** 2 + (cL[1] - c0[1]) ** 2) ** 0.5
  return disp, inst, len(frames)


def build(cfg: HybridConfig, cache_dir: Path, n: int, clip_arg: str | None) -> None:
  davis_dir = ENGINE_ROOT / "data_raw" / "DAVIS"
  jpeg_root, anno_root = _davis_roots(cfg, davis_dir)
  size = cfg.img_size
  available = {d.name for d in jpeg_root.iterdir() if d.is_dir()}

  # pick the clip whose frame-0 object travels FARTHEST (best-separated targets) unless given
  if clip_arg:
    clip = clip_arg
    info = _clip_travel(clip, jpeg_root, anno_root, size)
    if info is None:
      raise SystemExit(f"clip {clip} unusable")
    disp, inst, nframes = info
  else:
    scored = []
    for c in CURATED_PRISTINE:
      if c not in available:
        continue
      info = _clip_travel(c, jpeg_root, anno_root, size)
      if info:
        scored.append((info[0], c, info[1], info[2]))
    scored.sort(reverse=True)
    logger.info("clip travel ranking (top 6):")
    for d, c, _, nf in scored[:6]:
      logger.info(f"  {c:<16} travel={d:6.1f}px  frames={nf}")
    disp, clip, inst, nframes = scored[0]
  logger.info(f"OVERFIT clip = {clip}  (frame-0 object travels {disp:.1f}px over {nframes} frames)")

  frames = sorted((jpeg_root / clip).glob("*.jpg"))
  masks = sorted((anno_root / clip).glob("*.png"))

  # source = frame 0; targets = n frames evenly spaced from ~20% to the last frame
  src_idx = 0
  tgt_idxs = [int(round(x)) for x in np.linspace(nframes * 0.2, nframes - 1, n)]
  tgt_idxs = sorted(set(i for i in tgt_idxs if i > src_idx))
  logger.info(f"source frame {src_idx}, target frames {tgt_idxs}")

  backbone = DiffusionBackbone(cfg); backbone.to(cfg.device)
  (cache_dir / "latents").mkdir(parents=True, exist_ok=True)
  (cache_dir / "masks").mkdir(parents=True, exist_ok=True)
  manifest = {}

  m0 = _read_mask_indices(masks[src_idx])
  c0 = _centroid_scaled(m0, inst, size)
  src_frame = _load_frame(frames[src_idx], size).unsqueeze(0).to(cfg.device)
  src_norm = (src_frame * 2 - 1).to(cfg.compute_dtype)
  src_lat = backbone.encode(src_norm)[0].cpu()
  src_mask = _instance_mask_resized(m0, inst, size)

  with torch.no_grad():
    for ti in tgt_idxs:
      mt = _read_mask_indices(masks[ti])
      ct = _centroid_scaled(mt, inst, size)
      if ct is None:
        continue
      key = f"{clip}_overfit_{ti:05d}"
      tgt_frame = _load_frame(frames[ti], size).unsqueeze(0).to(cfg.device)
      tgt_norm = (tgt_frame * 2 - 1).to(cfg.compute_dtype)
      tgt_lat = backbone.encode(tgt_norm)[0].cpu()
      tgt_mask = _instance_mask_resized(mt, inst, size)

      torch.save({"source": src_lat, "target": tgt_lat}, cache_dir / "latents" / f"{key}.pt")
      np.save(cache_dir / "masks" / f"{key}.npy", src_mask)        # source mask (model input)
      np.save(cache_dir / "masks" / f"{key}_t.npy", tgt_mask)      # target mask (loss union)
      tx = (ct[0] - c0[0]) / size # type: ignore
      ty = (ct[1] - c0[1]) / size # type: ignore
      manifest[key] = {"trajectory": [tx, ty], "prompt_xy": [c0[0], c0[1]], # type: ignore
                       "target_xy": [ct[0], ct[1]]}
      logger.info(f"  {key}: traj=({tx:+.3f},{ty:+.3f})  target_px=({ct[0]:.0f},{ct[1]:.0f})")

  (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
  logger.info(f"OVERFIT cache: {len(manifest)} samples -> {cache_dir}")
  logger.info("source is IDENTICAL across all samples; only trajectory+target differ.")
  logger.info("Train hard (--max-steps 4000 --no-resume), then sample the TRAINED trajectories.")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser()
  p.add_argument("--cache-dir", type=Path, default=ENGINE_ROOT / "overfit_cache")
  p.add_argument("--n", type=int, default=8, help="number of target frames (samples)")
  p.add_argument("--clip", type=str, default=None, help="force a clip; default = farthest-travel")
  args = p.parse_args()
  cfg = HybridConfig()
  build(cfg, args.cache_dir, args.n, args.clip)


if __name__ == "__main__":
  main()