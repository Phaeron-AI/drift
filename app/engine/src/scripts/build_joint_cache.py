from __future__ import annotations

# =============================================================================
# T3 -- JOINT isolation test: BOTH pathways active at once.
#
# T1 (build_overfit_cache) proved trajectory->position with source FIXED.
# T2 (build_recon_cache)   proved source->appearance with trajectory ZERO.
# Neither exercised BOTH. Pathways can interfere -- trajectory may corrupt appearance, or the
# source may pin the object in place and fight the trajectory. T3 tests cooperation:
#
#   - N samples, each a DIFFERENT source (forces real appearance use)
#   - each with a real NON-ZERO trajectory and a real moving target
#   - pass = the SOURCE object (correct identity) appears at the TRAJECTORY-correct position
#
# This is the smallest test where success means "the full mechanism works." Only after T3 passes
# is a full-data run justified (failure there = data/capacity, not a broken pathway).
#
#   python -m engine.src.scripts.build_joint_cache --cache-dir joint_cache --n 12
#   ... train --cache-dir joint_cache --max-steps 4000 --batch-size 4 --accum-steps 2 --no-resume
#   then sample each source with ITS trained trajectory; check identity AND position together.
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

logger = logging.getLogger("drift.joint")
ENGINE_ROOT = Path(__file__).resolve().parent.parent


def _best_pair(clip, jpeg_root, anno_root, size, min_disp=25.0):
  """frame 0 as source; pick the target frame with the LARGEST clean displacement."""
  frames = sorted((jpeg_root / clip).glob("*.jpg"))
  masks = sorted((anno_root / clip).glob("*.png"))
  if len(frames) != len(masks) or len(frames) < 6:
    return None
  m0 = _read_mask_indices(masks[0]); inst = _pick_instance(m0)
  if inst is None: return None
  c0 = _centroid_scaled(m0, inst, size)
  if c0 is None: return None
  best = None
  for j in range(3, len(frames)):
    cj = _centroid_scaled(_read_mask_indices(masks[j]), inst, size)
    if cj is None: continue
    disp = ((cj[0]-c0[0])**2 + (cj[1]-c0[1])**2) ** 0.5
    if disp >= min_disp and (best is None or disp > best[1]):
      best = (j, disp, cj)
  if best is None: return None
  j, disp, cj = best
  return {"inst": inst, "c0": c0, "cj": cj, "src_i": 0, "tgt_j": j,
          "frames": frames, "masks": masks}


def build(cfg: HybridConfig, cache_dir: Path, n: int) -> None:
  davis_dir = ENGINE_ROOT / "data_raw" / "DAVIS"
  jpeg_root, anno_root = _davis_roots(cfg, davis_dir)
  size = cfg.img_size
  available = {d.name for d in jpeg_root.iterdir() if d.is_dir()}
  clips = [c for c in CURATED_PRISTINE if c in available]

  backbone = DiffusionBackbone(cfg); backbone.to(cfg.device)
  (cache_dir / "latents").mkdir(parents=True, exist_ok=True)
  (cache_dir / "masks").mkdir(parents=True, exist_ok=True)
  manifest = {}
  built = 0

  with torch.no_grad():
    for clip in clips:
      if built >= n: break
      pair = _best_pair(clip, jpeg_root, anno_root, size)
      if pair is None:
        continue
      inst, c0, cj = pair["inst"], pair["c0"], pair["cj"]
      fr, mk, i, j = pair["frames"], pair["masks"], pair["src_i"], pair["tgt_j"]

      src = (_load_frame(fr[i], size).unsqueeze(0).to(cfg.device) * 2 - 1).to(cfg.compute_dtype)
      tgt = (_load_frame(fr[j], size).unsqueeze(0).to(cfg.device) * 2 - 1).to(cfg.compute_dtype)
      src_lat = backbone.encode(src)[0].cpu()
      tgt_lat = backbone.encode(tgt)[0].cpu()
      m0 = _read_mask_indices(mk[i]); mj = _read_mask_indices(mk[j])

      key = f"{clip}_joint_00000"
      torch.save({"source": src_lat, "target": tgt_lat}, cache_dir / "latents" / f"{key}.pt")
      np.save(cache_dir / "masks" / f"{key}.npy", _instance_mask_resized(m0, inst, size))
      np.save(cache_dir / "masks" / f"{key}_t.npy", _instance_mask_resized(mj, inst, size))
      tx, ty = (cj[0]-c0[0])/size, (cj[1]-c0[1])/size
      manifest[key] = {"trajectory": [tx, ty], "prompt_xy": [c0[0], c0[1]],
                       "target_xy": [cj[0], cj[1]], "split": "train"}
      logger.info(f"  {key}: grab=({c0[0]:.0f},{c0[1]:.0f}) traj=({tx:+.3f},{ty:+.3f}) "
                  f"target=({cj[0]:.0f},{cj[1]:.0f})")
      built += 1

  (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
  logger.info(f"JOINT cache: {built} samples (distinct source + real trajectory each) -> {cache_dir}")
  logger.info("PASS = source identity preserved AND object at trajectory-correct position, together.")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser()
  p.add_argument("--cache-dir", type=Path, default=ENGINE_ROOT / "joint_cache")
  p.add_argument("--n", type=int, default=12)
  args = p.parse_args()
  cfg = HybridConfig()
  build(cfg, args.cache_dir, args.n)


if __name__ == "__main__":
  main()