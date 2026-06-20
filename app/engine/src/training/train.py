from __future__ import annotations

import re
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import diffusers.utils.logging as df_logging

from ..config import HybridConfig
from ..models.hybrid import HybridSpatialDiffusion
from .trainer import Trainer

logger = logging.getLogger('drift.train')
ENGINE_ROOT = Path(__file__).resolve().parent.parent


class CachedTripleDataset(Dataset):
  def __init__(self, cache_dir: Path, split: Optional[str] = None) -> None:
    self.cache_dir = cache_dir
    manifest = json.loads((cache_dir / "manifest.json").read_text())

    self.keys = [
      k for k in manifest
      if (cache_dir / "latents" / f"{k}.pt").exists()
      and (cache_dir / "masks" / f"{k}.npy").exists()
      # CLIP-LEVEL split: training must NEVER see val clips. split=None -> all (back-compat).
      and (split is None or manifest[k].get("split", "train") == split)
    ]

    self.manifest = manifest
    if not self.keys:
      raise RuntimeError(f"No usable cached samples in {cache_dir} (split={split}). Run precompute first.")
    logger.info(f"Dataset: {len(self.keys)} cached samples in {cache_dir} (split={split or 'all'})")

  def __len__(self) -> int:
    return len(self.keys)

  def _load_mask(self, key: str, suffix: str) -> Optional[torch.Tensor]:
    p = self.cache_dir / "masks" / f"{key}{suffix}.npy"
    if not p.exists():
      return None
    m = np.load(p)
    return torch.from_numpy(m).float().unsqueeze(0) / 255.0    # [1,H,W] in [0,1]

  def __getitem__(self, index: int) -> dict:
    key = self.keys[index]
    latents = torch.load(self.cache_dir / "latents" / f"{key}.pt", weights_only=True)

    # SOURCE mask -> model input (matches inference, which only has the source mask).
    mask_src = self._load_mask(key, "")                       # [1,H,W]
    # TARGET mask -> for the loss union. Fall back to source if absent (old/synthetic cache).
    mask_tgt = self._load_mask(key, "_t")
    if mask_tgt is None:
      mask_tgt = mask_src
    # UNION: object at start OR end. Rewards drawing the object at its destination AND
    # penalizes a ghost copy left at the origin. LOSS ONLY -- never fed to the model.
    loss_mask = torch.maximum(mask_src, mask_tgt) # type: ignore

    traj = torch.tensor(self.manifest[key]["trajectory"], dtype=torch.float32)

    return {
      "source": latents["source"],
      "target": latents["target"],
      "mask": mask_src,         # model input (attention bias) -- source position
      "loss_mask": loss_mask,   # loss weight -- union of source+target positions
      "trajectory": traj,
    }


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
  if not ckpt_dir.exists():
    return None
  best_path, best_step = None, -1
  for p in ckpt_dir.glob("step_*.pt"):
    m = re.fullmatch(r"step_(\d+)\.pt", p.name)
    if m:
      step = int(m.group(1))
      if step > best_step:
        best_step, best_path = step, p
  return best_path


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  df_logging.set_verbosity_error()

  parser = argparse.ArgumentParser(description="Train the spatial dynamics engine.")
  parser.add_argument("--cache-dir", type=Path, default=ENGINE_ROOT / "cache")
  parser.add_argument("--max-steps", type=int, default=20)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--batch-size", type=int, default=1, help="real batch dim; >1 smooths the loss")
  parser.add_argument("--accum-steps", type=int, default=8)
  parser.add_argument("--mask-weight", type=float, default=4.0)
  parser.add_argument("--ckpt-dir", type=Path, default=ENGINE_ROOT / "checkpoints")
  parser.add_argument("--ckpt-every", type=int, default=500)
  parser.add_argument("--resume-from", type=Path, default=None)
  parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and start fresh (disables auto-resume).")
  parser.add_argument("--split", type=str, default="train", help="which split to train on (train/val); train never sees held-out val clips")
  args = parser.parse_args()

  cfg = HybridConfig()
  logger.info(f"Device: {cfg.device} | compute_dtype: {cfg.compute_dtype} | img_size: {cfg.img_size}")

  model = HybridSpatialDiffusion(cfg)
  dataset = CachedTripleDataset(args.cache_dir, split=args.split)

  trainer = Trainer(
    cfg, model, dataset,
    lr=args.lr,
    batch_size=args.batch_size,
    accum_steps=args.accum_steps,
    mask_weight=args.mask_weight,
    ckpt_dir=args.ckpt_dir,
    ckpt_every=args.ckpt_every,
  )

  resume_from = args.resume_from
  if resume_from is None and not args.no_resume:
    latest = find_latest_checkpoint(args.ckpt_dir)
    if latest is not None:
      logger.info(f"Auto-resuming from latest checkpoint: {latest}")
      resume_from = latest
    else:
      logger.info("No checkpoint found; starting fresh.")
  elif args.no_resume:
    logger.info("--no-resume set; starting fresh (ignoring any checkpoints).")

  trainer.train(max_steps=args.max_steps, resume_from=resume_from)


if __name__ == "__main__":
  main()

"""
skate-park']
breakdance           grab=(197,264)  n=14
breakdance-flare     grab=(258,246)  n=11
car-roundabout       grab=(273,285)  n=2
dance-jump           grab=(270,240)  n=8
dogs-jump            grab=(239,256)  n=10
drift-straight       grab=(382,245)  n=8
judo                 grab=(278,228)  n=5
kite-walk            grab=(288,247)  n=10
libby                grab=(256,278)  n=4
motocross-bumps      grab=(193,281)  n=10
skate-park           grab=(315,297)  n=14
"""