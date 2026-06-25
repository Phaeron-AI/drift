from __future__ import annotations

import numpy as np
from PIL import Image as PILImage, ImageOps

from ..types import Image, Mask

def load_image(path: str) -> Image:

  pil_img = PILImage.open(path)
  pil_img = ImageOps.exif_transpose(pil_img)
  
  arr = np.array(pil_img)

  if arr.ndim == 2:
    arr = np.stack([arr, arr, arr], axis=-1)

  if arr.shape[-1] == 4:
    arr = arr[..., :3]
      
  if arr.shape[-1] != 3:
    raise ValueError(f"Image from {path} has unsupported shape {arr.shape} after processing.")

  return arr.astype(np.float64) / 255.0

def load_mask(path: str) -> Mask:
  pil_img = PILImage.open(path)
  pil_img = ImageOps.exif_transpose(pil_img)

  pil_img = pil_img.convert("L")
  
  arr = np.array(pil_img)

  return arr.astype(np.float64) / 255.0


def to_u8(img: Image) -> np.ndarray:
  return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)