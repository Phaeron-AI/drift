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

  # Models and Checkpoints (TBD)
  sam2_config_path: str = ""
  sam2_checkpoint_path: str = ""
  vae_model_id: str = ""
  unet_model_id: str = ""
  ddpm_model_id: str = ""

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

