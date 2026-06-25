from __future__ import annotations

import numpy as np

from ..src.config import RenderConfig
from ..src.core.composite import feather
from ..src.core.pipeline import loop_closure_error, render_cinemagraph
from ..src.types import Cinemagraph, Image, MotionEstimator, MotionField, Region, Segmenter


class RenderService:
  def __init__(self, segmenter: Segmenter, estimator: MotionEstimator, config: RenderConfig):
    self._segmenter = segmenter
    self._estimator = estimator
    self._cfg = config

  def render(self, image: Image) -> Cinemagraph:
    regions = self._segmenter.segment(image)
    M = self._combine_fields(image, regions)
    alpha = feather(self._animate_mask(regions), sigma=self._cfg.feather_sigma)
    frames = render_cinemagraph(image, alpha, M, n_frames=self._cfg.n_frames)
    err = loop_closure_error(image, alpha, M, n_frames=self._cfg.n_frames)
    return Cinemagraph(frames=frames, fps=self._cfg.fps, seamless=err < 1e-6)

  def _combine_fields(self, image: Image, regions: list[Region]) -> MotionField:
    h, w = image.shape[:2]
    M = np.zeros((h, w, 2), dtype=np.float64)
    for r in regions:
      M = M + self._estimator.estimate(image, r)   # disjoint masks -> sum places each
    return M

  def _animate_mask(self, regions: list[Region]) -> np.ndarray:
    h, w = regions[0].mask.shape
    acc = np.zeros((h, w), dtype=np.float64)
    for r in regions:
      if not r.is_locked:
        acc = np.clip(acc + r.mask, 0.0, 1.0)
    return acc