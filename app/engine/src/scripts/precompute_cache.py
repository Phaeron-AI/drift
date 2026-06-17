from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TF

from ..config import HybridConfig
from ..src.models.backbones import DiffusionBackbone, FrozenSam2

logger = logging.getLogger("drift.precompute_cache")


def _sample_key(sample: dict) -> str:
  if "id" in sample:
    return str(sample["id"])
  raw = Path(sample["source_image_path"])
  return f"{raw.parent.name}_{raw.stem}"


def _load_source_target_images(sample: dict, cfg: HybridConfig) -> tuple[Tensor, Tensor]:
  if "source_image" in sample:
    src = sample["source_image"]
    tgt = sample["target_image"]
    if src.ndim == 3: src = src.unsqueeze(0)
    if tgt.ndim == 3: tgt = tgt.unsqueeze(0)
    src = src.to(cfg.device); tgt = tgt.to(cfg.device)
  else:
    from PIL import Image
    src_pil = Image.open(sample["source_image_path"]).convert("RGB").resize(
      (cfg.img_size, cfg.img_size), Image.Resampling.BILINEAR)
    tgt_pil = Image.open(sample["target_image_path"]).convert("RGB").resize(
      (cfg.img_size, cfg.img_size), Image.Resampling.BILINEAR)
    src = TF.to_tensor(src_pil).unsqueeze(0).to(cfg.device)
    tgt = TF.to_tensor(tgt_pil).unsqueeze(0).to(cfg.device)

  src_norm = (src * 2.0) - 1.0      # VAE expects [-1, 1]
  tgt_norm = (tgt * 2.0) - 1.0
  return src_norm.to(cfg.compute_dtype), tgt_norm.to(cfg.compute_dtype)


def _grab_point(sample: dict) -> tuple[float, float]:
  return float(sample["start_x"]), float(sample["start_y"])


def _trajectory_vec(sample: dict, cfg: HybridConfig) -> tuple[float, float]:
  dx = sample["end_x"] - sample["start_x"]
  dy = sample["end_y"] - sample["start_y"]
  return float(dx / cfg.img_size), float(dy / cfg.img_size)


def build_dataset(cfg: HybridConfig) -> Iterator[dict]:
  # TODO[dataset]: chain your real sources here. e.g.:
  yield {
    "source_image_path": "dummy_src.jpg", "target_image_path": "dummy_tgt.jpg",
    "start_x": 100, "start_y": 150, "end_x": 120, "end_y": 170,
  }


def precompute(cfg: HybridConfig, cache_dir: Path, overwrite: bool) -> None:
  (cache_dir / "latents").mkdir(parents=True, exist_ok=True)
  (cache_dir / "masks").mkdir(parents=True, exist_ok=True)
  manifest_path = cache_dir / "manifest.json"
  manifest: dict[str, dict] = {}
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())

  backbone = DiffusionBackbone(cfg)        
  backbone.to(cfg.device)
  sam2 = FrozenSam2(dataclasses.replace(cfg, sam2_residency="resident"))

  written, skipped = 0, 0
  for sample in build_dataset(cfg):
    try:
      key = _sample_key(sample)
      lat_path = cache_dir / "latents" / f"{key}.pt"
      mask_path = cache_dir / "masks" / f"{key}.npy"
      if lat_path.exists() and mask_path.exists() and not overwrite:
        skipped += 1
        continue

      source_img, target_img = _load_source_target_images(sample, cfg)
      x, y = _grab_point(sample)
      prompt = torch.tensor([[x, y]], device=cfg.device, dtype=torch.float32)

      src_lat = backbone.encode(source_img)[0].cpu()    # [4,h,w] scaled latent
      tgt_lat = backbone.encode(target_img)[0].cpu()
      sam_img = (source_img + 1.0) / 2.0                # SAM2 wants [0,1]
      mask = sam2.predict_mask(sam_img, prompt)         # [1,1,H,W]
      mask_np = (mask[0, 0] > 0.5).to(torch.uint8).cpu().numpy() * 255

      torch.save({"source": src_lat, "target": tgt_lat}, lat_path)
      np.save(mask_path, mask_np)

      tx, ty = _trajectory_vec(sample, cfg)
      manifest[key] = {"trajectory": [tx, ty], "prompt_xy": [float(x), float(y)]}
      written += 1
      if written % 256 == 0:
        manifest_path.write_text(json.dumps(manifest))
        logger.info(f"checkpoint: {written} cached...")
    except Exception as e:
      logger.error(f"Failed sample {sample.get('id', sample.get('source_image_path', '?'))}: {e}")
      continue

  manifest_path.write_text(json.dumps(manifest))
  logger.info(f"Precompute complete. written={written} skipped={skipped} -> {cache_dir}")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  parser = argparse.ArgumentParser(description="Precompute source/target latents + masks.")
  parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()
  cfg = HybridConfig()
  with torch.no_grad():
    precompute(cfg, args.cache_dir, args.overwrite)


if __name__ == "__main__":
  main()