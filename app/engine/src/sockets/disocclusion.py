from __future__ import annotations

import numpy as np

from ..types import Image, Mask

class PlateFillDisocclusion:
  def fill(self, warped: Image, plate: Image, hole_alpha: Mask) -> Image:
    a = hole_alpha[..., None]
    return (1.0 - a) * warped + a * plate
  

class DiffusionDisocclusion:
  def __init__(self, checkpoint: str, device: str = "cuda"):
    self.device = device
    
    import torch
    try:
      from diffusers import StableDiffusionInpaintPipeline  # type: ignore
    except ImportError:
      raise ImportError(
        "DiffusionDisocclusion requires the 'diffusers' and 'transformers' packages. "
        "Install them via: pip install diffusers transformers"
      )
    try:
      self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint, 
        torch_dtype=torch.float16,
        variant="fp16",
        requires_safety_checker=False
      ).to(self.device)
      self.pipe.set_progress_bar_config(disable=True) 
    except Exception as e:
      raise RuntimeError(f"Failed to load diffusion inpainting model from {checkpoint}. Error: {e}")

  def fill(self, warped: Image, plate: Image, hole_alpha: Mask) -> Image:
    import torch
    from PIL import Image as PILImage

    warp_uint8 = (np.clip(warped, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_uint8 = (np.clip(hole_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    warp_pil = PILImage.fromarray(warp_uint8)
    mask_pil = PILImage.fromarray(mask_uint8)

    with torch.no_grad():
      output = self.pipe(
        prompt="",
        image=warp_pil,
        mask_image=mask_pil,
        num_inference_steps=20,
        output_type="np" 
      ).images[0]  # Shape: (H, W, 3) # type: ignore

    a = hole_alpha[..., None]

    final_image = (1.0 - a) * warped + a * output
    
    return final_image