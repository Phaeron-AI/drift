from __future__ import annotations

import os

import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter

from .src.core.composite import feather
from .src.core.fields import apply_region, sway_field
from .src.core.pipeline import loop_closure_error, render_cinemagraph

OUT_DIR = os.environ.get("OUT_DIR", "./out")
H, W = 360, 320
N_FRAMES = 48


def build_scene() -> tuple[np.ndarray, np.ndarray]:
  yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
  strands = 0.5 + 0.32 * np.sin(2 * np.pi * xx / 9.0)
  coarse = gaussian_filter(np.random.default_rng(7).random((H, W)), 18)
  coarse = (coarse - coarse.min()) / (np.ptp(coarse) + 1e-9)
  bg_lum = np.clip(0.35 + 0.55 * strands * (0.6 + 0.4 * coarse), 0, 1)
  sky = np.clip(1.0 - yy / H, 0, 1)[..., None]
  foliage = np.stack([bg_lum * 0.45, bg_lum * 0.85, bg_lum * 0.40], axis=-1)
  image = foliage * (0.75 + 0.25 * (1 - sky)) + np.array([0.55, 0.62, 0.70]) * (0.25 * sky)
  image = np.clip(image, 0, 1)
  cy, cx = 0.46 * H, 0.5 * W
  head = ((yy - cy) ** 2) / (62.0 ** 2) + ((xx - cx) ** 2) / (50.0 ** 2) <= 1.0
  shoulders = (yy > cy + 40) & (
      ((xx - cx) ** 2) / (130.0 ** 2) + ((yy - (cy + 150)) ** 2) / (150.0 ** 2) <= 1.0)
  subject = head | shoulders
  image[subject] = np.array([0.86, 0.74, 0.66])
  region = (~subject).astype(np.float64)
  return image, region


def to_u8(img: np.ndarray) -> np.ndarray:
  return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main() -> None:
  os.makedirs(OUT_DIR, exist_ok=True)
  image, region = build_scene()
  M = apply_region(sway_field(H, W, amplitude=4.5, wavelength=140.0), region)
  alpha = feather(region, sigma=5.0)
  err = loop_closure_error(image, alpha, M, N_FRAMES)
  print(f"loop-closure error: {err:.2e}  ({'seamless' if err < 1e-6 else 'CHECK'})")
  frames = render_cinemagraph(image, alpha, M, N_FRAMES)
  print(f"frame 0 vs source: {float(np.max(np.abs(frames[0]-image))):.2e}")
  imageio.imwrite(os.path.join(OUT_DIR, "scene_input.png"), to_u8(image))
  imageio.mimsave(os.path.join(OUT_DIR, "cinemagraph.gif"),
                  [to_u8(f) for f in frames], duration=1 / 24, loop=0)
  print(f"wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
  main()