from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..config import HybridConfig
from ..models.backbones import DiffusionBackbone, FrozenSam2
from ..data.davis_adapter import build_davis_dataset, DavisConfig
from ..data.movi_adapter import build_movi_dataset, MoviConfig

logger = logging.getLogger("drift.precompute_cache")
ENGINE_ROOT = Path(__file__).resolve().parent.parent


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

  src_norm = (src * 2.0) - 1.0
  tgt_norm = (tgt * 2.0) - 1.0
  return src_norm.to(cfg.compute_dtype), tgt_norm.to(cfg.compute_dtype)


def _grab_point(sample: dict, which: str) -> tuple[float, float]:
  if which == "source":
    return float(sample["start_x"]), float(sample["start_y"])
  return float(sample["end_x"]), float(sample["end_y"])


def _trajectory_vec(sample: dict, cfg: HybridConfig) -> tuple[float, float]:
  dx = sample["end_x"] - sample["start_x"]
  dy = sample["end_y"] - sample["start_y"]
  return float(dx / cfg.img_size), float(dy / cfg.img_size)


def _sam_mask(sam2: FrozenSam2, img_pm1: Tensor, x: float, y: float, cfg: HybridConfig) -> np.ndarray:
  # FALLBACK only -- used when a sample has no ground-truth mask (e.g. the video extractor).
  sam_img = (img_pm1 + 1.0) / 2.0
  prompt = torch.tensor([[x, y]], device=cfg.device, dtype=torch.float32)
  m = sam2.predict_mask(sam_img, prompt)
  return (m[0, 0] > 0.5).to(torch.uint8).cpu().numpy() * 255


def build_dataset(cfg: HybridConfig, args) -> Iterator[dict]:
  # SOURCE dispatch. Each branch yields the SAME triple dict shape, and each tags its own
  # "split" (DAVIS: clip-level holdout; MOVi: test->val), which precompute() stamps verbatim
  # into the manifest. So train/val separation is the adapter's job, not the cache's.
  if args.source == "davis":
    davis_dir = ENGINE_ROOT / "data_raw" / "DAVIS"
    # FULL-DAVIS de-risking run: every clip, dense sampling, 12% of clips held out as val.
    dcfg = DavisConfig(use_all_clips=True, holdout_frac=0.12, holdout_seed=0)
    yield from build_davis_dataset(cfg, davis_dir, dcfg)
  elif args.source == "movi":
    # MOVi-C: GROUND-TRUTH masks + trajectory (no SAM). split="train" -> train; split="test" ->
    # the adapter tags those samples "val" (held-out objects AND backgrounds = honest eval).
    # Run this script twice into the SAME --cache-dir (--split train, then --split test); the
    # movic_{split}_... keys are disjoint so the two passes coexist in one manifest.
    mcfg = MoviConfig(split=args.split, seed=args.seed, max_videos=args.max_videos)
    yield from build_movi_dataset(cfg, mcfg)
  else:
    raise ValueError(f"unknown --source {args.source!r} (expected 'davis' or 'movi')")


def precompute(cfg: HybridConfig, args) -> None:
  cache_dir: Path = args.cache_dir
  overwrite: bool = args.overwrite
  (cache_dir / "latents").mkdir(parents=True, exist_ok=True)
  (cache_dir / "masks").mkdir(parents=True, exist_ok=True)
  manifest_path = cache_dir / "manifest.json"
  manifest: dict[str, dict] = {}
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())

  backbone = DiffusionBackbone(cfg)
  backbone.to(cfg.device)
  sam2: Optional[FrozenSam2] = None   # built lazily ONLY if a sample lacks GT masks

  written, skipped = 0, 0
  for sample in build_dataset(cfg, args):
    try:
      key = _sample_key(sample)
      lat_path = cache_dir / "latents" / f"{key}.pt"
      mask_path = cache_dir / "masks" / f"{key}.npy"        # SOURCE mask (model input)
      mask_t_path = cache_dir / "masks" / f"{key}_t.npy"    # TARGET mask (loss union)
      if lat_path.exists() and mask_path.exists() and mask_t_path.exists() and not overwrite:
        # Files already cached -> skip the (expensive) re-encode. But if a PRIOR run crashed
        # between writing these files and the periodic manifest flush, the entry is missing and
        # the sample would be orphaned (invisible to training). Reconstruct it from the sample in
        # hand -- no re-encode, no network -- so resume is lossless.
        if key not in manifest:
          sx, sy = _grab_point(sample, "source")
          tx, ty = _trajectory_vec(sample, cfg)
          manifest[key] = {"trajectory": [tx, ty], "prompt_xy": [float(sx), float(sy)],
                           "split": sample.get("split", "train")}
        skipped += 1
        continue

      source_img, target_img = _load_source_target_images(sample, cfg)
      src_lat = backbone.encode(source_img)[0].cpu()
      tgt_lat = backbone.encode(target_img)[0].cpu()

      if "source_mask" in sample and "target_mask" in sample:
        # GROUND-TRUTH masks from the adapter -- whole-object, no fragments (P-006 fix).
        mask_s_np = sample["source_mask"]
        mask_t_np = sample["target_mask"]
      else:
        # FALLBACK: no GT masks (non-DAVIS/non-MOVi source) -> SAM at the grab points.
        if sam2 is None:
          sam2 = FrozenSam2(dataclasses.replace(cfg, sam2_residency="resident"))
        sx, sy = _grab_point(sample, "source")
        ex, ey = _grab_point(sample, "target")
        mask_s_np = _sam_mask(sam2, source_img, sx, sy, cfg)
        mask_t_np = _sam_mask(sam2, target_img, ex, ey, cfg)

      torch.save({"source": src_lat, "target": tgt_lat}, lat_path)
      np.save(mask_path, mask_s_np)
      np.save(mask_t_path, mask_t_np)

      sx, sy = _grab_point(sample, "source")
      tx, ty = _trajectory_vec(sample, cfg)
      manifest[key] = {"trajectory": [tx, ty], "prompt_xy": [float(sx), float(sy)],
                       "split": sample.get("split", "train")}
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
  parser.add_argument("--cache-dir", type=Path, default=ENGINE_ROOT / "cache")
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument("--source", choices=["davis", "movi"], default="davis",
                      help="dataset adapter to precompute from")
  parser.add_argument("--split", type=str, default="train",
                      help="MOVi only: 'train' or 'test' (test is tagged split=val in the manifest)")
  parser.add_argument("--max-videos", type=int, default=None,
                      help="MOVi only: cap number of videos streamed (None = whole split, ~9.75k train)")
  parser.add_argument("--seed", type=int, default=0,
                      help="MOVi only: pair-selection RNG seed (reproducible cache)")
  args = parser.parse_args()
  cfg = HybridConfig()
  with torch.no_grad():
    precompute(cfg, args)


if __name__ == "__main__":
  main()