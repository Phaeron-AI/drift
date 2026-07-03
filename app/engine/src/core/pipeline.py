from __future__ import annotations

import numpy as np

from .composite import composite
from .integration import integrate_series
from .warp import backward_warp


def render_cinemagraph(image: np.ndarray, alpha: np.ndarray, M: np.ndarray, n_frames: int = 48) -> list[np.ndarray]:
  plate = image
  disp_neg = integrate_series(M, n_frames, sign=-1.0)   # dest->source (forward advection)
  disp_pos = integrate_series(M, n_frames, sign=+1.0)   # homeward half of the loop

  frames: list[np.ndarray] = []
  for t in range(n_frames):
    w = t / n_frames
    layer_from_start = backward_warp(image, disp_neg[t])
    layer_toward_end = backward_warp(image, disp_pos[n_frames - t])
    moving = (1.0 - w) * layer_from_start + w * layer_toward_end
    frames.append(composite(moving, plate, alpha))
  return frames


def loop_closure_error(image: np.ndarray, alpha: np.ndarray, M: np.ndarray, n_frames: int = 48) -> float:
  disp_pos = integrate_series(M, n_frames, sign=+1.0)
  implied_frame_N = composite(backward_warp(image, disp_pos[0]), image, alpha)
  frame_0 = composite(backward_warp(image, integrate_series(M, n_frames, -1.0)[0]), image, alpha)
  return float(np.max(np.abs(implied_frame_N - frame_0)))