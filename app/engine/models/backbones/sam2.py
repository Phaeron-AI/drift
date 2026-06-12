from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from ...config import HybridConfig

logger = logging.getLogger("drift.segment_anything")


class FrozenSam2(nn.Module):
  def __init__(self, cfg: HybridConfig) -> None:
    super().__init__()
    self.cfg = cfg

    self.residency_device = (
      torch.device("cpu")
      if cfg.sam2_residency == "offload"
      else torch.device(cfg.device)
    )

    logger.info(
      f"Loading SAM2 from {cfg.sam2_checkpoint_path} to {self.residency_device}"
    )
    
    model = build_sam2(
      cfg.sam2_config_path,
      cfg.sam2_checkpoint_path,
      device=self.residency_device, # type: ignore
    )
    self.predictor = SAM2ImagePredictor(model)

    self.predictor.model.to(dtype=cfg.compute_dtype)

    self._freeze()

  def _freeze(self) -> None:
    for param in self.predictor.model.parameters():
      param.requires_grad_(False)
    self.predictor.model.eval()
    logger.info("SAM2 frozen (requires_grad=False + eval()).")

  @torch.no_grad()
  def predict_mask(self, image: Tensor, prompt_points: Tensor) -> Tensor:
    batch_size, _, height, width = image.shape
    batch_masks = []

    if self.cfg.sam2_residency == "offload":
      self.predictor.model.to(torch.device(self.cfg.device))

    try:
      for i in range(batch_size): 
        arr = image[i].permute(1, 2, 0).cpu().float().numpy() * 255.0
        img_np = arr.clip(0, 255).astype(np.uint8)

        self.predictor.set_image(img_np)

        pt_coordinate = prompt_points[i].cpu().numpy().reshape(1, 2)
        pt_label = np.array([1])

        masks, scores, _ = self.predictor.predict(
          point_coords=pt_coordinate,
          point_labels=pt_label,
          multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)]        
        best_mask_tensor = (
          torch.from_numpy(best_mask)
          .unsqueeze(0)                         
          .to(torch.device(self.cfg.device))
        )
        batch_masks.append(best_mask_tensor)
    finally:
      if self.cfg.sam2_residency == "offload":
        self.predictor.model.to(torch.device("cpu"))

    final_mask_tensor = torch.stack(batch_masks).float()   

    if final_mask_tensor.shape[-2:] != (height, width):
      final_mask_tensor = F.interpolate(
        final_mask_tensor, size=(height, width), mode="nearest"
      )

    return final_mask_tensor