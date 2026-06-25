from __future__ import annotations

import numpy as np
import warnings

from ..core.fields import apply_region, sway_field, zero_field
from ..types import Image, MotionField, Region

class AnalyticMotionEstimator:
  def __init__(self, amplitude_px: float = 4.5, wavelength_px: float = 140.0)-> None:
    self._amp = amplitude_px
    self._wave = wavelength_px
  
  def estimate(self, image: Image, region: Region)-> MotionField:
    h, w = region.mask.shape
    if region.is_locked or region.strength == 0.0:
      return zero_field(h, w)
    M = sway_field(h, w, amplitude=self._amp * region.strength, wavelength=self._wave)
    return apply_region(M, region.mask)

class LearnedMotionEstimator:
  def __init__(self, checkpoint: str, device: str = "cuda", div_threshold: float = 0.5) -> None:
    self.device = device
    self.div_threshold = div_threshold

    import torch

    try:
      self.model = torch.jit.load(checkpoint, map_location=self.device)
      self.model.save()
    except Exception as e:
      raise RuntimeError(f"Failed to load motion model from {checkpoint}. Ensure it's a valid PyTorch model format. Error: {e}")
    
  def estimate(self, image: Image, region: Region)-> MotionField:
    import torch

    h, w = region.mask.shape
    if region.is_locked or region.strength == 0.0:
      return zero_field(h, w)
    
    img_t = torch.from_numpy(image).float().permute(2, 0, 1)  # (3, H, W)
    mask_t = torch.from_numpy(region.mask).float().unsqueeze(0)  # (1, H, W)
    x = torch.cat([img_t, mask_t], dim=0).unsqueeze(0).to(self.device)

    with torch.no_grad():
      pred_t = self.model(x)
    
    M = pred_t.squeeze(0).cpu().numpy()

    M = M * region.strength

    dy_dy, _ = np.gradient(M[0], axis=(0, 1))
    _, dx_dx = np.gradient(M[1], axis=(0, 1))

    divergence = dx_dx + dy_dy
    max_div = np.max(divergence)

    if max_div > self.div_threshold:
      scale_factor = self.div_threshold / max_div
      warnings.warn(
        f"Region '{region.name}': Large divergence detected ({max_div:.3f} > {self.div_threshold}). "
        f"Down-scaling field by {scale_factor:.2f}x to prevent disocclusion tearing."
      )
      M = M * scale_factor
        
    return apply_region(M, region.mask)