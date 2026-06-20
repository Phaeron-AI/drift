from __future__ import annotations

# =============================================================================
# Joint-test (T3) evaluator. Samples EVERY clip in the joint cache using its EXACT trained
# trajectory (read from the manifest), so the test is honest -- no stand-in dx/dy that could
# exaggerate or mask the result. For each clip it reports grab point + trained trajectory and
# saves a source|generated panel. Read each: is it the RIGHT OBJECT (identity) AND at the
# RIGHT POSITION (trajectory)? Both must hold for T3 to pass.
#
#   python -m engine.src.scripts.joint_eval --checkpoint engine/src/checkpoints/step_4000.pt
# =============================================================================

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from ..config import HybridConfig
from .sample_inference import sample          # reuse the SAME tested inference path

logger = logging.getLogger("drift.joint_eval")
ENGINE_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--cache-dir", type=Path, default=ENGINE_ROOT / "joint_cache")
  p.add_argument("--davis-root", type=str,
                 default=str(ENGINE_ROOT / "data_raw" / "DAVIS" / "JPEGImages" / "480p"))
  p.add_argument("--out-dir", type=str, default="viz/joint_eval")
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--n", type=int, default=99, help="max clips to evaluate")
  args = p.parse_args()

  cfg = HybridConfig()
  manifest = json.loads((args.cache_dir / "manifest.json").read_text())
  davis_root = Path(args.davis_root)
  out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

  done = 0
  for key, v in manifest.items():
    if done >= args.n:
      break
    clip = key.replace("_joint_00000", "")
    gx, gy = v["prompt_xy"]
    tx, ty = v["trajectory"]
    dx, dy = tx * cfg.img_size, ty * cfg.img_size       # EXACT trained trajectory, in pixels
    src = davis_root / clip / "00000.jpg"
    if not src.exists():
      logger.warning(f"{clip}: source frame not found at {src}; skipping")
      continue

    logger.info(f"{clip}: grab=({gx:.0f},{gy:.0f}) trained dx={dx:+.0f} dy={dy:+.0f}")
    sample_args = SimpleNamespace(
      source=str(src), checkpoint=args.checkpoint,
      grab_x=gx, grab_y=gy, dx=dx, dy=dy, steps=args.steps,
      out=str(out_dir / f"joint_{clip}.png"),
    )
    try:
      sample(cfg, sample_args)
      done += 1
    except Exception as e:
      logger.error(f"{clip}: sampling failed: {e}")

  logger.info(f"joint_eval: {done} panels -> {out_dir}")
  logger.info("PASS criterion per panel: correct OBJECT IDENTITY *and* trajectory-correct POSITION.")


if __name__ == "__main__":
  main()