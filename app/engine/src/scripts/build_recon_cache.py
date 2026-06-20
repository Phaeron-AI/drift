from __future__ import annotations

# =============================================================================
# RECONSTRUCTION overfit cache -- the dual of build_overfit_cache.
#
# build_overfit_cache held SOURCE constant and varied TRAJECTORY -> isolated the trajectory
# pathway (but could pass WITHOUT using the source at all, which is why it never validated
# appearance). This does the opposite:
#
#   - N DIFFERENT sources (one per clip's frame 0)
#   - target = source  (identity reconstruction)
#   - trajectory = (0,0)  (no motion)
#
# The ONLY way to fit all N is to COPY source -> output through the source-conditioning channels.
# The model cannot cheat: sources differ (can't memorize one output), trajectory is zero
# everywhere (carries no info). This is the cleanest possible test of "does the source pathway
# carry appearance?" -- the thing we wrongly assumed the trajectory overfit had proven.
#
# Run:   python -m engine.src.scripts.build_recon_cache --cache-dir recon_cache --n 10
# Train: ... train --cache-dir recon_cache --max-steps 3000 --batch-size 4 --accum-steps 2 --no-resume
# Test:  sample each source back with --dx 0 --dy 0 ; output should REPRODUCE the source.
# =============================================================================

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from ..config import HybridConfig
from ..models.backbones import DiffusionBackbone
from ..data.davis_adapter import (
  _davis_roots, _read_mask_indices, _pick_instance, _centroid_scaled,
  _instance_mask_resized, _load_frame, CURATED_PRISTINE,
)

logger = logging.getLogger("drift.recon")
ENGINE_ROOT = Path(__file__).resolve().parent.parent


def build(cfg: HybridConfig, cache_dir: Path, n: int) -> None:
  davis_dir = ENGINE_ROOT / "data_raw" / "DAVIS"
  jpeg_root, anno_root = _davis_roots(cfg, davis_dir)
  size = cfg.img_size
  available = {d.name for d in jpeg_root.iterdir() if d.is_dir()}
  clips = [c for c in CURATED_PRISTINE if c in available][:n]
  logger.info(f"RECON cache from {len(clips)} clips (one identity sample each): {clips}")

  backbone = DiffusionBackbone(cfg); backbone.to(cfg.device)
  (cache_dir / "latents").mkdir(parents=True, exist_ok=True)
  (cache_dir / "masks").mkdir(parents=True, exist_ok=True)
  manifest = {}

  with torch.no_grad():
    for clip in clips:
      frames = sorted((jpeg_root / clip).glob("*.jpg"))
      masks = sorted((anno_root / clip).glob("*.png"))
      if not frames:
        continue
      m0 = _read_mask_indices(masks[0])
      inst = _pick_instance(m0)
      if inst is None:
        continue
      c0 = _centroid_scaled(m0, inst, size)
      if c0 is None:
        continue

      frame = _load_frame(frames[0], size).unsqueeze(0).to(cfg.device)
      norm = (frame * 2 - 1).to(cfg.compute_dtype)
      lat = backbone.encode(norm)[0].cpu()
      msk = _instance_mask_resized(m0, inst, size)

      key = f"{clip}_recon_00000"
      # target latent == source latent (identity); both masks identical; trajectory zero.
      torch.save({"source": lat, "target": lat}, cache_dir / "latents" / f"{key}.pt")
      np.save(cache_dir / "masks" / f"{key}.npy", msk)
      np.save(cache_dir / "masks" / f"{key}_t.npy", msk)
      manifest[key] = {"trajectory": [0.0, 0.0], "prompt_xy": [c0[0], c0[1]], "split": "train"}
      logger.info(f"  {key}: grab=({c0[0]:.0f},{c0[1]:.0f})  (target=source, traj=0)")

  (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
  logger.info(f"RECON cache: {len(manifest)} identity samples -> {cache_dir}")
  logger.info("Train hard, then sample each source with --dx 0 --dy 0; output must REPRODUCE it.")
  logger.info("If it reproduces the different sources -> source pathway WORKS (data scale was the")
  logger.info("issue). If it can't -> source pathway is the root cause (architectural).")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser()
  p.add_argument("--cache-dir", type=Path, default=ENGINE_ROOT / "recon_cache")
  p.add_argument("--n", type=int, default=10, help="number of distinct-source identity samples")
  args = p.parse_args()
  cfg = HybridConfig()
  build(cfg, args.cache_dir, args.n)


if __name__ == "__main__":
  main()