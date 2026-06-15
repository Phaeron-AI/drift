from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import HybridConfig
from ..models.hybrid import HybridSpatialDiffusion
from ..utils.vram import VRAMManager
from ..losses import l1_l2_reconstruction_loss, temporal_consistency_loss

logger = logging.getLogger("drift.trainer")

class Trainer:
  def __ini__(
    self, cfg: HybridConfig, 
    model: HybridSpatialDiffusion, 
    dataset, 
    lr: float = 1e-4, 
    accum_steps: int = 8, 
    mask_weight: float = 4.0, 
    temporal_weight: float = 0.0, 
    ckpt_dir: Path = Path("checkpoints"), 
    ckpt_every: int = 500
  )-> None:
    self.cfg = cfg
    self.model = model
    self.accum_steps = accum_steps
    self.mask_weight = mask_weight
    self.temporal_weight = temporal_weight
    self.ckpt_dir = ckpt_dir
    self.ckpt_every = ckpt_every
    self.global_step = 0
    self.vram = VRAMManager(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    self.loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    self.optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
  
  def _noise_batch(self, target_latent)-> tuple[Tensor, ...]:
    B = target_latent.shape[0]
    noise = torch.randn_like(target_latent)
    t = torch.randint(0, self.model.backbone.scheduler.config.num_train_timesteps, (B,), device=target_latent.device).long()  # type: ignore
    noisy = self.model.backbone.scheduler.add_noise(target_latent, noise, t)  # type: ignore

    return noisy, noise, t
  
  def _step(self, batch)-> float:
    device = self.cfg.device
    compute_dtype = self.cfg.compute_dtype

    source_latent = batch["source"].to(device, compute_dtype)   # [1,4,h,w] clean reference
    target_latent = batch["target"].to(device, compute_dtype)   # [1,4,h,w] Object to denoise
    mask = batch["mask"].to(device) # [1,1,H,W]
    trajectory = batch["trajectory"].to(device, compute_dtype)  # [1, 2] normalized
    text_embeds = batch.get("text_embeds", None)

    if text_embeds is not None:
      text_embeds = text_embeds.to(device, compute_dtype)

    with torch.autocast(device_type="cuda", dtype=self.cfg.compute_dtype):
      noisy, noise, t = self._noise_batch(target_latent)
      noise_pred = self.model(noisy, t, trajectory, mask, source_latent, text_embeds)
      
      recon = l1_l2_reconstruction_loss(noise_pred, noise, mask, self.mask_weight)
      temporal = temporal_consistency_loss(noise_pred.unsqueeze(1), noise.unsqueeze(1))
      
      loss = recon + self.temporal_weight * temporal

    scaled_loss = loss / self.accum_steps
    scaled_loss.backward()

    if (self.global_step + 1) % self.accum_steps == 0:
      self.optimizer.step()
      self.optimizer.zero_grad(set_to_none=True)

    return loss.detach().item()

  
  def train(self, max_steps: int, resume_from: Optional[Path] = None)-> None:
    if resume_from is not None:
      self.load_checkpoint(resume_from)
    
    self.model.train()

    VRAMManager.report("train start")

    done = False
    while not done:
      for batch in self.loader:
        loss_val = self._step(batch)
 
        if self.global_step % 50 == 0:
          self.vram.report(f"step {self.global_step}")
          logger.info(f"step {self.global_step} | loss {loss_val:.4f}")
 
        self.global_step += 1
        if self.global_step % self.ckpt_every == 0:
          self.save_checkpoint()
        if self.global_step >= max_steps:
          done = True
          break
 
    self.save_checkpoint()
    logger.info(f"Training complete at step {self.global_step}.")
  
  def save_checkpoint(self)-> None:
    path = self.ckpt_dir / f"step_{self.global_step}.pt"
    torch.save({"step": self.global_step, "trainable": {k: v for k, v in self.model.state_dict().items() if self.model.get_parameter(k).requires_grad}}, path)
  
  def load_checkpoint(self, path: Path)-> None:
    if not path.exists():
      logger.warning(f"Checkpoint not found at {path}. Starting from scratch.")
      return

    logger.info(f"Loading checkpoint from {path}...")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if "model_state_dict" in checkpoint:
      missing_keys, unexpected_keys = self.model.load_state_dict(
        checkpoint["model_state_dict"], 
        strict=False
      )
      if unexpected_keys:
        logger.warning(f"Found {len(unexpected_keys)} unexpected keys in checkpoint.")
      logger.info(f"Model weights restored successfully.")
    else:
      logger.error("Checkpoint file is missing 'model_state_dict'.")
    
    if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      logger.info("Optimizer state restored.")

    self.global_step = checkpoint.get("global_step", 0)

    if hasattr(self, "epoch"):
      self.epoch = checkpoint.get("epoch", 0)

    logger.info(f"Resumed training state -> Step: {self.global_step}")