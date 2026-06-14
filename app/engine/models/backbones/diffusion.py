from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler  # type: ignore[import]
from peft import LoraConfig, get_peft_model

from ...config import HybridConfig

logger = logging.getLogger("spatial_dynamics.diffusion")


class DiffusionBackbone(nn.Module):
  def __init__(self, cfg: HybridConfig) -> None:
    super().__init__()
    self.cfg = cfg

    logger.info(f"Loading VAE: {cfg.vae_model_id}")
    self.vae = AutoencoderKL.from_pretrained(
      cfg.vae_model_id,
      subfolder="vae",
      torch_dtype=cfg.vae_dtype,
    )

    logger.info(f"Loading U-Net: {cfg.unet_model_id}")
    self.unet = UNet2DConditionModel.from_pretrained(
      cfg.unet_model_id,
      subfolder="unet",
      torch_dtype=cfg.compute_dtype,
    )

    logger.info(f"Loading scheduler: {cfg.ddpm_model_id}")
    self.scheduler = DDPMScheduler.from_pretrained(
      cfg.ddpm_model_id,
      subfolder="scheduler",
    )

    if hasattr(self.vae.config, "scaling_factor"):
      self.vae_scaling_factor = self.vae.config.scaling_factor  # type: ignore
    else:
      self.vae_scaling_factor = 0.18215  # SD 1.5/2.1 default
      logger.warning(
        f"VAE scaling_factor missing; defaulting to {self.vae_scaling_factor}"
      )

    self._freeze()
    if cfg.enable_gradient_checkpointing:
      self._enable_gradient_checkpointing()
    self._inject_lora()

  def _freeze(self) -> None:
    self.vae.requires_grad_(False)
    self.unet.requires_grad_(False)
    self.vae.eval()
    self.unet.eval()
    logger.info("Base VAE and U-Net frozen (requires_grad=False + eval()).")

  def _enable_gradient_checkpointing(self) -> None:
    self.unet.enable_gradient_checkpointing()  # type: ignore
    logger.info("Gradient checkpointing enabled on U-Net.")

  def _inject_lora(self) -> None:
    lora_config = LoraConfig(
      r=self.cfg.lora_rank,
      lora_alpha=self.cfg.lora_alpha,
      lora_dropout=self.cfg.lora_dropout,
      target_modules=list(self.cfg.lora_target_modules),
      init_lora_weights="gaussian",
    )

    self.unet = get_peft_model(self.unet, lora_config)  # type: ignore

    trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in self.unet.parameters() if not p.requires_grad)
    pct = 100.0 * trainable / max(trainable + frozen, 1)
    logger.info(f"LoRA injected. Trainable: {trainable:,} ({pct:.3f}%) | Frozen: {frozen:,}")

  @torch.no_grad()
  def encode(self, pixels: Tensor) -> Tensor:
    pixels = pixels.to(dtype=self.cfg.vae_dtype)
    posterior = self.vae.encode(pixels).latent_dist  # type: ignore

    latents = posterior.sample()
    latents = latents * self.vae_scaling_factor

    return latents.to(dtype=self.cfg.compute_dtype)

  @torch.no_grad()
  def decode(self, latents: Tensor) -> Tensor:
    latents = latents / self.vae_scaling_factor
    latents = latents.to(dtype=self.cfg.vae_dtype)
    image = self.vae.decode(latents).sample  # type: ignore
    return image
  
  def _expand_conv_in(self) -> None:
    old = self.unet.conv_in   # raw UNet here (call this BEFORE _inject_lora): Conv2d(4, 320, 3, padding=1)

    new = nn.Conv2d(in_channels=8, out_channels=old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=(old.bias is not None)) # type: ignore

    with torch.no_grad():
      new.weight.zero_()                   # all channels start at 0
      new.weight[:, :4].copy_(old.weight)  # channels 0:4 = pretrained noisy-latent path; 4:8 stay zero # type: ignore
      if old.bias is not None:  # type: ignore
        new.bias.copy_(old.bias)  # type: ignore

    new = new.to(device=old.weight.device, dtype=old.weight.dtype)  # type: ignore
    self.unet.conv_in = new 

  def _set_conv_in_trainable(self) -> None:
    unet = self.unet.base_model.model if hasattr(self.unet, "base_model") else self.unet
    for p in unet.conv_in.parameters(): # type: ignore
      p.requires_grad_(True)

    n = sum(p.numel() for p in unet.conv_in.parameters() if p.requires_grad)  # type: ignore
    logger.info(f"conv_in expanded to 8ch and set trainable: {n:,} params")