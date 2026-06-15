"""
Run:  python -m engine.data.visualize_pairs --video path/to/clip.mp4 --n 10 --out viz/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ..config import HybridConfig
from ..models.backbones import FrozenSam2
from .video_pair_extractor import extract_pairs, PairConfig
import dataclasses


def _to_pil(frame_chw: torch.Tensor) -> Image.Image:
  arr = (frame_chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
  return Image.fromarray(arr)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--video", type=str, required=True)
  p.add_argument("--n", type=int, default=10)
  p.add_argument("--out", type=Path, default=Path("viz"))
  args = p.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  cfg = HybridConfig()
  sam2 = FrozenSam2(dataclasses.replace(cfg, sam2_residency="resident"))

  count = 0
  for triple in extract_pairs(args.video, sam2, cfg, PairConfig()):
    src = _to_pil(triple["source_image"]); tgt = _to_pil(triple["target_image"])
    W, H = src.size
    canvas = Image.new("RGB", (W * 2 + 10, H), (0, 0, 0))
    canvas.paste(src, (0, 0)); canvas.paste(tgt, (W + 10, 0))
    d = ImageDraw.Draw(canvas)
    sx, sy, ex, ey = triple["start_x"], triple["start_y"], triple["end_x"], triple["end_y"]
    d.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], outline=(0, 255, 0), width=2)        # start on source
    d.line([sx, sy, ex, ey], fill=(255, 255, 0), width=2)                            # drag vector
    d.ellipse([W + 10 + ex - 4, ey - 4, W + 10 + ex + 4, ey + 4], outline=(255, 0, 0), width=2)  # end on target
    canvas.save(args.out / f"{triple['id']}.png")
    count += 1
    if count >= args.n:
      break
  print(f"wrote {count} visualizations to {args.out}")


if __name__ == "__main__":
  main()