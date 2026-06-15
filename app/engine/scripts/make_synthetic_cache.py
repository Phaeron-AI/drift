from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch

from ..config import HybridConfig


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--cache-dir", type=Path, default=Path("cache"))
  p.add_argument("--n", type=int, default=8)
  args = p.parse_args()

  cfg = HybridConfig()
  h = cfg.latent_size
  (args.cache_dir / "latents").mkdir(parents=True, exist_ok=True)
  (args.cache_dir / "masks").mkdir(parents=True, exist_ok=True)
  manifest = {}

  for i in range(args.n):
    key = f"synthetic_{i:03d}"
    torch.save(
      {"source": torch.randn(4, h, h), "target": torch.randn(4, h, h)},
      args.cache_dir / "latents" / f"{key}.pt",
    )
    # a centered square mask, uint8 [H,W]
    m = np.zeros((cfg.img_size, cfg.img_size), dtype=np.uint8)
    q = cfg.img_size // 4
    m[q:3*q, q:3*q] = 255
    np.save(args.cache_dir / "masks" / f"{key}.npy", m)
    manifest[key] = {"trajectory": [0.1, -0.05], "prompt_xy": [256.0, 256.0]}

  (args.cache_dir / "manifest.json").write_text(json.dumps(manifest))
  print(f"wrote {args.n} synthetic samples to {args.cache_dir}")


if __name__ == "__main__":
  main()