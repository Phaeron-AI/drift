from __future__ import annotations

# =============================================================================
# THE VERDICT -- held-out A/B controllability test on MOVi-C `val` (held-out objects AND
# backgrounds). For each held-out source, run the SAME source with OPPOSITE trajectories
# (+dx and -dx). PASS = the object lands on opposite sides AND keeps its identity (the same
# GSO object), on data the model has never seen. This is the test the whole synthetic-data
# thesis rides on: clean data at scale (MOVi) vs the 742-sample DAVIS wall.
#
# Why this script exists separately from joint_eval: joint_eval points sample_inference at a
# DAVIS .jpg on disk. MOVi `val` frames are inside the tfds stream -- not on disk. So we decode
# the cached SOURCE latent (exactly what the model conditions on) back to a PNG, then feed it
# through the SAME tested sample() path everything else uses. No new inference code.
#
# Precondition: a movi cache with split=val entries AND a trained checkpoint:
#   python -m engine.src.scripts.precompute_cache --source movi --split test --cache-dir engine/src/movi_cache --max-videos 200
#   python -m engine.src.training.train --cache-dir engine/src/movi_cache --split train ... (see sequence)
#
# Run:
#   python -m engine.src.scripts.movi_ab --checkpoint engine/src/checkpoints_movi/step_16000.pt \
#       --cache-dir engine/src/movi_cache --k 6 --disp 140
# =============================================================================

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ..config import HybridConfig
from ..models.backbones import DiffusionBackbone
from .sample_inference import sample          # reuse the SAME tested inference path

logger = logging.getLogger("drift.movi_ab")
ENGINE_ROOT = Path(__file__).resolve().parent.parent


def _export_source_png(backbone: DiffusionBackbone, cache_dir: Path, key: str,
                       out_path: Path, cfg: HybridConfig) -> None:
  # Decode the cached SOURCE latent -> PNG. This is the appearance the model actually conditions
  # on (one VAE round-trip), so it's the honest source for an A/B about controllability.
  lat = torch.load(cache_dir / "latents" / f"{key}.pt", weights_only=True)["source"]
  lat = lat.unsqueeze(0).to(cfg.device, cfg.compute_dtype)
  img = backbone.decode(lat)                                 # [1,3,H,W] ~[-1,1]
  img = (img.float().clamp(-1, 1) + 1.0) / 2.0
  arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
  Image.fromarray(arr).save(out_path)


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser(description="Held-out A/B controllability test on MOVi-C val.")
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--cache-dir", type=Path, default=ENGINE_ROOT / "movi_cache")
  p.add_argument("--out-dir", type=str, default="viz/movi_ab")
  p.add_argument("--k", type=int, default=6, help="number of held-out sources to A/B")
  p.add_argument("--disp", type=float, default=140.0, help="+/- x displacement (px) for the A/B")
  p.add_argument("--steps", type=int, default=50)
  args = p.parse_args()

  cfg = HybridConfig()
  manifest = json.loads((args.cache_dir / "manifest.json").read_text())
  all_val = [k for k, v in manifest.items() if v.get("split") == "val"]
  if not all_val:
    raise SystemExit(
      f"no split=val entries in {args.cache_dir}/manifest.json -- run "
      "`precompute_cache --source movi --split test` into this cache first."
    )
  val_keys = all_val[: args.k]
  logger.info(f"A/B on {len(val_keys)} of {len(all_val)} held-out (val) sources")

  out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
  src_dir = out_dir / "_sources"; src_dir.mkdir(exist_ok=True)

  # Build a backbone ONCE only to decode cached source latents -> PNG, then release it before
  # sampling (sample() builds its own full model; keep peak to one model at a time on 12GB).
  backbone = DiffusionBackbone(cfg); backbone.to(cfg.device)
  png_paths: dict[str, Path] = {}
  with torch.no_grad():
    for key in val_keys:
      pth = src_dir / f"{key}.png"
      _export_source_png(backbone, args.cache_dir, key, pth, cfg)
      png_paths[key] = pth
  del backbone
  torch.cuda.empty_cache()

  done = 0
  for key in val_keys:
    gx, gy = manifest[key]["prompt_xy"]
    for tag, dx in (("right", +args.disp), ("left", -args.disp)):
      sa = SimpleNamespace(
        source=str(png_paths[key]), checkpoint=args.checkpoint,
        grab_x=float(gx), grab_y=float(gy), dx=float(dx), dy=0.0,
        steps=args.steps, out=str(out_dir / f"{key}_{tag}.png"),
      )
      try:
        sample(cfg, sa)
        done += 1
      except Exception as e:
        logger.error(f"{key} {tag}: sampling failed: {e}")

  logger.info(f"movi_ab: {done} panels -> {out_dir}")
  logger.info("PASS = for each held-out source, object moves RIGHT under +dx and LEFT under -dx,")
  logger.info("       with the SAME object identity preserved (unseen object + unseen background).")
  logger.info("If yes: thesis confirmed (architecture was right, data was the wall).")


if __name__ == "__main__":
  main()