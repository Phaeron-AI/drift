from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ...config.schema import HybridConfig
from .backbones import DiffusionBackbone, FrozenSam2
from .attention import TrajectoryEncoder, install_spatial_processors

logger = logging.getLogger("drift.hybrid")


class HybridSpatialDiffusion(nn.Module):
  def __init__(self, cfg: HybridConfig, n_traj: int = 1, target_latent_sizes: tuple[int, ...] = (8, 16, 32)) -> None:
    super().__init__()
    self.cfg = cfg
    self.n_traj = n_traj

    self.backbone = DiffusionBackbone(cfg)
    self.traj_encoder = TrajectoryEncoder(embed_dim=768, n_tokens=n_traj)

    self.mask_biases = install_spatial_processors(   
      self.backbone.unet,
      self.traj_encoder,
      target_latent_sizes=target_latent_sizes,
      n_traj=n_traj,
    )

    self.to(cfg.device)
    self.traj_encoder.to(dtype=cfg.compute_dtype)
    self.mask_biases.to(dtype=cfg.compute_dtype)

    self._log_trainable()

  def _log_trainable(self) -> None:
    n = sum(p.numel() for p in self.trainable_parameters())
    logger.info(f"HybridSpatialDiffusion trainable params: {n:,} "f"(LoRA + TrajectoryEncoder + MaskToBias)")

  def trainable_parameters(self) -> Iterator[nn.Parameter]:
    return (p for p in self.parameters() if p.requires_grad)

  def encode(self, pixels: Tensor) -> Tensor:
    return self.backbone.encode(pixels)

  def decode(self, latents: Tensor) -> Tensor:
    return self.backbone.decode(latents)

  def forward(
    self,
    noisy_latents: Tensor,           # [B, C, h, w]
    timestep: Tensor,                # [B]
    trajectory: Tensor,              # [B, 2]   (normalized to [-1, 1])
    mask: Tensor,                    # [B, 1, H, W]
    source_latent: Tensor,
    text_embeds: Optional[Tensor] = None,
  ) -> Tensor:
    if text_embeds is None:
      batch_size = noisy_latents.shape[0]
      text_embeds = torch.zeros(
        (batch_size, 77, 768),
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
      )

    unet_input = torch.cat([noisy_latents, source_latent], dim=1) 

    noise_pred = self.backbone.unet(
      unet_input,
      timestep,
      encoder_hidden_states=text_embeds,
      cross_attention_kwargs={"trajectory": trajectory, "mask": mask},
    ).sample

    return noise_pred

  @torch.no_grad()
  def predict_mask(self, image: Tensor, prompt_points: Tensor) -> Tensor:
    if not hasattr(self, "_sam2"):
      logger.info("Initializing FrozenSam2 lazily for inference pass.")
      self._sam2 = FrozenSam2(self.cfg)
      self._sam2.to(self.cfg.device)

    return self._sam2.predict_mask(image, prompt_points)