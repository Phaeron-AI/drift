from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger("drift.phase1")

@dataclass
class HybridConfig:
  # Device Settings
  device: str = "cuda"
  compute_dtype: torch.dtype = torch.bfloat16
  vae_dtype: torch.dtype = torch.bfloat16
  torch.backends.cudnn.enabled = False

  # Models and Checkpoints
  sam2_config_path: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
  sam2_checkpoint_path: str = "engine/models/saved/segment_anything/sam2.1_hiera_base_plus.pt"
  vae_model_id: str = "runwayml/stable-diffusion-v1-5"
  unet_model_id: str = "runwayml/stable-diffusion-v1-5"
  ddpm_model_id: str = "runwayml/stable-diffusion-v1-5"

  # Spatial Dimensions
  img_size: int = 512
  vae_downsample_dim: int = 8
  latent_channels: int = 4

  sam2_residency: Literal["cache", "offload", "resident"] = "offload"
  enable_gradient_checkpointing: bool = True

  # LoRA
  lora_rank: int = 8
  lora_alpha: int = 16
  lora_dropout: float = 0.0

  lora_target_modules: tuple[str, ...] = field(
    default_factory=lambda: ("to_q", "to_k", "to_v", "to_out.0")
  )

  @property
  def latent_size(self)-> int:
    return self.img_size // self.vae_downsample_dim

def verify_environment(cfg: HybridConfig) -> None:
  torch.backends.cudnn.enabled = False
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable — check your torch install / driver.")

  cap = torch.cuda.get_device_capability(0)
  if cap != (12, 0):
    raise RuntimeError(f"Expected sm_120 (12, 0); got {cap}.")

  if "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError(
      f"torch built without Blackwell kernels. arch_list={torch.cuda.get_arch_list()}. "
      "Reinstall from the cu128/cu130 index."
    )

  if not torch.cuda.is_bf16_supported():
    raise RuntimeError("bfloat16 not supported on this device.")

  from torch.nn.attention import SDPBackend, sdpa_kernel
  q = torch.randn(1, 8, 128, 64, device=cfg.device, dtype=cfg.compute_dtype)
  try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
      torch.nn.functional.scaled_dot_product_attention(q, q, q)
    logger.info("SDPA flash backend OK")
  except RuntimeError:
    logger.warning("Flash SDPA unavailable — falling back to math/efficient kernel.")

  free_b, total_b = torch.cuda.mem_get_info()
  logger.info(f"GPU free:  {free_b  / 1024**3:.2f} GiB")
  logger.info(f"GPU total: {total_b / 1024**3:.2f} GiB")
  logger.info(f"Device: {torch.cuda.get_device_name(0)}  capability={cap}")
