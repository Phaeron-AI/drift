from __future__ import annotations

import copy
import numpy as np

from ..types import Cinemagraph
from .image_io import to_u8


def save_gif(cine: Cinemagraph, path: str) -> None:
  try:
    import imageio
  except ImportError:
    raise ImportError("The 'imageio' package is required for exporting GIFs. Run: pip install imageio")

  u8_frames = [to_u8(frame) for frame in cine.frames]

  imageio.mimsave(path, u8_frames, format='GIF', duration=1.0 / cine.fps, loop=0) # type: ignore


def save_mp4(cine: Cinemagraph, path: str) -> None:
  try:
    import imageio
  except ImportError:
    raise ImportError("The 'imageio[ffmpeg]' package is required for exporting MP4s. Run: pip install imageio[ffmpeg]")

  u8_frames = [to_u8(frame) for frame in cine.frames]

  imageio.mimsave(path,  u8_frames,  format='FFMPEG',  fps=cine.fps,  codec='libx264',  macro_block_size=16)  # type: ignore


def apply_watermark(cine: Cinemagraph, *, tier: str) -> Cinemagraph:
  """Stamps a watermark on trial exports without polluting the renderer."""
  # HINT: Pro tier skips this completely
  if tier.lower() == "pro":
      return cine

  from PIL import Image as PILImage, ImageDraw

  marked_cine = copy.copy(cine)
  marked_frames = []

  for frame in cine.frames:
    img_u8 = to_u8(frame)
    pil_img = PILImage.fromarray(img_u8)
    draw = ImageDraw.Draw(pil_img)

    width, height = pil_img.size
    box_w, box_h = 100, 25
    x, y = width - box_w - 15, height - box_h - 15
    
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0))
    draw.text((x + 8, y + 6), "TRIAL VERSION", fill=(255, 255, 255))

    marked_frames.append(np.array(pil_img).astype(np.float64) / 255.0)

  marked_cine.frames = marked_frames
  return marked_cine