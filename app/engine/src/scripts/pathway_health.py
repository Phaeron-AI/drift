from __future__ import annotations

# =============================================================================
# Pathway health check -- run on ANY checkpoint (or fresh model) to catch a silently-dead
# conditioning pathway BEFORE wasting a training run. This is the single check that would have
# caught the zero-init source-pathway bug on run 1 instead of run 6.
#
#   python -m engine.src.scripts.pathway_health --checkpoint engine/src/checkpoints/step_3000.pt
#   python -m engine.src.scripts.pathway_health            # fresh model (check init)
#
# Reports each conditioning pathway's weight scale relative to the path it augments. A pathway
# that is >5-10x quieter than its reference is being IGNORED regardless of "trainable" status.
# =============================================================================

import argparse
import logging
from pathlib import Path

import torch

from ..config import HybridConfig
from ..models.hybrid import HybridSpatialDiffusion

logger = logging.getLogger("drift.health")


def _norm(t) -> float:
  return float(t.float().norm().item())


def check(model: HybridSpatialDiffusion) -> None:
  import torch.nn as nn
  sd = model.state_dict()
  rows = []

  # --- conv_in: source channels (4:8) vs noise channels (0:4) ---
  # Read off the LIVE 8-channel module via named_modules(), NOT the first "conv_in" tensor in the
  # state dict. After PEFT (LoRA) wrapping the state dict contains MULTIPLE conv_in-named 4D
  # tensors (base reference + wrapped), and grabbing [0] can hit a stale/zero one -> false "DEAD".
  # (This was a real false-positive: script said 0.000 while the model reconstructed bear/camel.
  #  Lesson: cross-check a diagnostic against a known result before trusting its verdict.)
  conv = None
  for name, mod in model.named_modules():
    if isinstance(mod, nn.Conv2d) and getattr(mod, "in_channels", None) == 8 and name.endswith("conv_in"):
      conv = mod
      break
  if conv is not None:
    w = conv.weight
    noise_n, src_n = _norm(w[:, :4]), _norm(w[:, 4:])
    ratio = src_n / noise_n if noise_n else 0.0
    verdict = "OK" if ratio >= 0.2 else "DEAD/QUIET -- source likely IGNORED"
    rows.append(("conv_in source vs noise", f"{src_n:.3f}", f"{noise_n:.3f}", f"{ratio:.3f}", verdict))

  # --- trajectory encoder: is it nonzero? ---
  traj_w = [v for k, v in sd.items() if "trajectory" in k.lower() and v.ndim >= 2]
  if traj_w:
    tn = max(_norm(v) for v in traj_w)
    rows.append(("trajectory encoder", f"{tn:.3f}", "-", "-", "OK" if tn > 1e-3 else "DEAD (all ~0)"))

  # --- mask-to-bias: zero-init by design; nonzero after training = it's being used ---
  mask_w = [v for k, v in sd.items() if "mask" in k.lower() and "to_bias" in k.lower() or
            ("MaskToBias".lower() in k.lower())]
  mask_w = [v for k, v in sd.items() if "mask" in k.lower() and v.ndim >= 2]
  if mask_w:
    mn = max(_norm(v) for v in mask_w)
    rows.append(("mask-to-bias", f"{mn:.3f}", "-", "-",
                 "active" if mn > 1e-3 else "still ~0 (unused or untrained)"))

  w = max(len(r[0]) for r in rows) if rows else 10
  logger.info(f"{'pathway':<{w}} | {'norm':>8} | {'ref':>8} | {'ratio':>6} | verdict")
  logger.info("-" * (w + 40))
  for name, n, ref, ratio, verdict in rows:
    logger.info(f"{name:<{w}} | {n:>8} | {ref:>8} | {ratio:>6} | {verdict}")

  bad = [r for r in rows if "DEAD" in r[4] or "IGNORED" in r[4]]
  if bad:
    logger.warning(f"{len(bad)} pathway(s) look silent -- isolation-test them before any full run.")
  else:
    logger.info("All pathways audible. (Audible != correct -- still run the isolation tests.)")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", type=Path, default=None, help="omit to check a FRESH model's init")
  args = p.parse_args()
  cfg = HybridConfig()
  model = HybridSpatialDiffusion(cfg)
  if args.checkpoint:
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    logger.info(f"loaded {args.checkpoint}")
  else:
    logger.info("checking FRESH model init (no checkpoint)")
  check(model)


if __name__ == "__main__":
  main()