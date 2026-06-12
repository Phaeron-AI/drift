from __future__ import annotations

import gc
import logging
import torch
import torch.nn as nn

from typing import List
from contextlib import contextmanager

from ..config import HybridConfig

logger = logging.getLogger("drift.vram")

class VRAMManager:
  def __init__(self, cfg: HybridConfig)-> None:
    super().__init__()
    self.cfg = cfg
  
  @staticmethod
  def report(tag: str = "")-> None:
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak  = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(f"[{tag}] allocated={alloc:.2f} GiB | peak={peak:.2f} GiB")
    torch.cuda.reset_peak_memory_stats()
  
  def budget(self, vae: nn.Module, unet: nn.Module)-> None:
    BYTES_BF16, BYTES_FP32 = 2, 4
    frozen_models: List = [vae, unet]
    
    frozen_params = sum(p.numel() for m in frozen_models for p in m.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for m in frozen_models for p in m.parameters() if p.requires_grad)
    weights = frozen_params * BYTES_BF16
    lora_master = trainable_params * BYTES_FP32
    gradients = trainable_params * BYTES_FP32
    optimizer = trainable_params * BYTES_FP32 * 2

    fixed_total = (weights + lora_master + gradients + optimizer) / 1024**3 # type: ignore
    logger.info(
      f"VRAM budget (excl. activations): {fixed_total:.2f} GiB  "
      f"[weights={weights/1024**3:.2f} | lora+grad+optim="  # type: ignore
      f"{(lora_master+gradients+optimizer)/1024**3:.3f}]"
    ) 
    logger.info("Activations are the remaining ~budget; profile with report().")

  @contextmanager
  def borrow_gpu(self, module: nn.Module):
    try:
      module.to(self.cfg.device)
      yield module
    finally:
      module.to("cpu")
      gc.collect()
      torch.cuda.empty_cache()