from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RenderConfig:
  n_frames: int = 48
  fps: float = 24.0
  amplitude_px: float = 4.5
  wavelength_px: float = 140.0
  feather_sigme: float = 5.0
  edge_mode: str = "reflect"
  device: str = "cpu"
  # seed: float = 42