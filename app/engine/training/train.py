from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
  def __init__(self, cache_dir: Path)-> None:
    self.cache_dir = cache_dir
    mainfest = json.loads((cache_dir / "manifest.json").read_text())

    self.keys = [
      k for k in mainfest
      if (cache_dir / "latents" / f"{k}.pt").exists()
      and (cache_dir / "masks" / f"{k}.npy").exists()
    ]

    self.manifest = mainfest
    if not self.keys:
      raise RuntimeError(f"No usable cached samples in {cache_dir}. Run precompute first.")
    logger.info(f"Dataset: {len(self.keys)} cached samples in {cache_dir}")

  def __len__(self)-> int:
    return len(self.keys)
  
  def __getitem__(self, index: int) -> dict:
    key = self.keys[index]
    latents = torch.load(self.cache_dir / "latents" / f"{key}.pt", weights_only=True)
    mask_np = np.load(self.cache_dir / "masks" / f"{key}.npy")  # uint8 [H,W] in {0,255}
    mask = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0        # [1,H,W] in [0,1]
    traj = torch.tensor(self.manifest[key]["trajectory"], dtype=torch.float32)  # [2]

    return {
      "source": latents["source"],
      "target": latents["target"],
      "mask": mask,
      "trajectory": traj,
    }
  
  
def main()-> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  df_logging.set_verbosity_error()   
 
  parser = argparse.ArgumentParser(description="Train the spatial dynamics engine.")
  parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
  parser.add_argument("--max-steps", type=int, default=20)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--accum-steps", type=int, default=8)
  parser.add_argument("--mask-weight", type=float, default=4.0)
  parser.add_argument("--ckpt-dir", type=Path, default=ENGINE_ROOT / "checkpoints")
  parser.add_argument("--ckpt-every", type=int, default=500)
  parser.add_argument("--resume-from", type=Path, default=None)
  args = parser.parse_args()
 
  cfg = HybridConfig()
  logger.info(f"Device: {cfg.device} | compute_dtype: {cfg.compute_dtype} | img_size: {cfg.img_size}")
 
  model = HybridSpatialDiffusion(cfg)
  dataset = CachedTripleDataset(args.cache_dir)
 
  trainer = Trainer(
    cfg, model, dataset,
    lr=args.lr,
    accum_steps=args.accum_steps,
    mask_weight=args.mask_weight,
    ckpt_dir=args.ckpt_dir,
    ckpt_every=args.ckpt_every,
  )
  trainer.train(max_steps=args.max_steps, resume_from=args.resume_from)

if __name__ == "__main__":
  main()