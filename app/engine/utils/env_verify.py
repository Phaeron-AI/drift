from __future__ import annotations

import logging

import torch

from ..config import HybridConfig, verify_environment
from ..models.backbones.sam2 import FrozenSam2
from ..models.backbones.diffusion import DiffusionBackbone
from ..utils.vram import VRAMManager

logger = logging.getLogger("spatial_dynamics.verify")


def _check_trainable_boundary(backbone: DiffusionBackbone) -> None:
  trainable, total, non_lora = 0, 0, []
  for name, p in backbone.unet.named_parameters():
    total += p.numel()
    if p.requires_grad:
      trainable += p.numel()
      if "lora_" not in name:
          non_lora.append(name)

  pct = 100.0 * trainable / max(total, 1)
  logger.info(f"CHECK 1  trainable={trainable:,} ({pct:.3f}% of U-Net)")

  assert not non_lora, (
    f"Non-LoRA trainable params found ({len(non_lora)}); freezing/targeting is wrong. "
    f"First few: {non_lora[:5]}"
  )
  assert pct < 1.5, (
    f"Trainable share {pct:.2f}% is too high -- target_modules likely over-matched. "
    "Expected <~1% for rank-r LoRA on the four attention projections."
  )
  logger.info("CHECK 1 PASSED: all trainable params are LoRA factors.")


def _check_sam2_frozen(sam2: FrozenSam2) -> None:
  leaks = [n for n, p in sam2.predictor.model.named_parameters() if p.requires_grad]
  assert not leaks, f"SAM2 has {len(leaks)} trainable params; _freeze() failed. {leaks[:5]}"
  logger.info("CHECK 1b PASSED: SAM2 fully frozen.")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  cfg = HybridConfig()

  verify_environment(cfg)

  vram = VRAMManager(cfg)
  VRAMManager.report("baseline")

  logger.info("Loading DiffusionBackbone...")
  backbone = DiffusionBackbone(cfg)
  VRAMManager.report("after diffusion load")

  logger.info("Loading FrozenSam2...")
  sam2 = FrozenSam2(cfg)
  VRAMManager.report("after sam2 load")

  _check_trainable_boundary(backbone)
  _check_sam2_frozen(sam2)

  backbone.to(cfg.device)
  vram.budget(backbone.vae, backbone.unet)

  B, H, W = 1, cfg.img_size, cfg.img_size
  image = torch.rand(B, 3, H, W, device=cfg.device, dtype=cfg.compute_dtype)
  prompt = torch.tensor([[W // 2, H // 2]], device=cfg.device, dtype=torch.float32)  # center grab

  VRAMManager.report("before round-trip")

  mask = sam2.predict_mask(image, prompt)               # [B,1,H,W]
  VRAMManager.report("after predict_mask")
  assert mask.shape == (B, 1, H, W), f"mask shape {mask.shape} != {(B,1,H,W)}"
  fg = mask.mean().item()
  logger.info(f"CHECK 3  mask foreground fraction = {fg:.3f}")

  assert 0.0 < fg < 1.0, (
    "Degenerate mask (all-fg or all-bg). Check the image-range conversion in sam2.py."
  )

  latents = backbone.encode(image)                      # [B,C,h,w]
  VRAMManager.report("after encode")
  assert latents.shape == (B, cfg.latent_channels, cfg.latent_size, cfg.latent_size), \
    f"latent shape {latents.shape} unexpected"

  recon = backbone.decode(latents)                      # [B,3,H,W]
  VRAMManager.report("after decode")
  assert recon.shape == image.shape, f"recon shape {recon.shape} != {image.shape}"
  assert torch.isfinite(recon).all(), (
    "Non-finite values in decode output -- the float16-VAE black-frame issue. "
    "Set vae_dtype to bfloat16 or float32."
  )

  logger.info("=" * 60)
  logger.info("PHASE 1 SMOKE TEST PASSED -- frozen boundary, budget, and round-trip OK.")
  logger.info("=" * 60)


if __name__ == "__main__":
  main()