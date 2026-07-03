"""
Run: python -m living_portraits.tests.test_core  (from the project root)
"""
from __future__ import annotations

import numpy as np

from ..src.core.fields import constant_field, rotational_field
from ..src.core.integration import integrate_series
from ..src.core.sampling import base_grid


def test_constant_field_is_exact() -> None:
  H, W, N = 64, 64, 30
  vy, vx = 0.37, -0.21
  M = constant_field(H, W, vy, vx)
  series = integrate_series(M, N, sign=1.0)
  max_err = 0.0
  for t in range(N + 1):
    expected = np.zeros((H, W, 2))
    expected[..., 0] = t * vy
    expected[..., 1] = t * vx
    max_err = max(max_err, float(np.max(np.abs(series[t] - expected))))
  assert max_err < 1e-9, f"constant-field integration not exact: {max_err:.2e}"
  print(f"[PASS] constant field exact to t·c   (max error {max_err:.2e})")


def test_rotational_field_curves_and_conserves_radius() -> None:
  H, W, N = 201, 201, 60
  omega = 0.03
  M = rotational_field(H, W, omega=omega)
  series = integrate_series(M, N, sign=1.0)
  grid = base_grid(H, W)
  cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
  py, px = 60, 140
  r0 = np.hypot(py - cy, px - cx)
  max_radius_drift = 0.0
  for t in range(N + 1):
    pos = grid[py, px] + series[t][py, px]
    r = np.hypot(pos[0] - cy, pos[1] - cx)
    max_radius_drift = max(max_radius_drift, abs(r - r0))
  linear_guess = grid[py, px] + N * M[py, px]
  integrated_end = grid[py, px] + series[N][py, px]
  linear_vs_integrated = float(np.hypot(*(linear_guess - integrated_end)))
  assert max_radius_drift < 0.06 * r0, (
    f"radius not conserved: drift {max_radius_drift:.2f} px on r0={r0:.1f}")
  assert linear_vs_integrated > 5.0, (
    "integrated path indistinguishable from straight line")
  print(f"[PASS] rotation curves & conserves radius "
    f"(radius drift {max_radius_drift:.2f}px / r0 {r0:.1f}px; "
    f"integrated vs linear {linear_vs_integrated:.1f}px apart)")


if __name__ == "__main__":
  test_constant_field_is_exact()
  test_rotational_field_curves_and_conserves_radius()
  print("\nAll integration-core tests passed.")