"""
Target: python -m living_portraits.cli portrait.jpg --mask subject.png -o out.gif
"""
from __future__ import annotations

import argparse
import sys
import time

try:
  from .src.io.image_io import load_image, load_mask
  from .src.sockets.segmentation import ManualMaskSegmentor
  from .src.sockets.motion_estimator import AnalyticMotionEstimator
  from .src.render_service import RenderService
  from .src.config import RenderConfig
  from .src.io.export import save_gif
except ImportError as e:
  try:
    from .src.io.image_io import load_image, load_mask
    from .src.sockets.segmentation import ManualMaskSegmentor
    from .src.sockets.motion_estimator import AnalyticMotionEstimator
    from .src.io.export import save_gif
    from .src.render_service import RenderService
    from .src.config import RenderConfig
  except ImportError:
    raise RuntimeError(f"Failed to resolve pipeline imports. Ensure you are executing as a module. Original error: {e}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Render a living-portrait cinemagraph.")
  parser.add_argument("image", help="Path to the input source image.")
  parser.add_argument("--mask", required=True, help="subject mask (1=locked, 0=animatable)")
  parser.add_argument("-o", "--out", default="cinemagraph.gif", help="Output file path.")
  parser.add_argument("--frames", type=int, default=48, help="Number of frames for the loop.")
  parser.add_argument("--amplitude", type=float, default=4.5, help="Amplitude of the sway effect in pixels.")
  args = parser.parse_args()

  print(f"[1/4] Loading assets...")
  print(f"Image: {args.image}")
  print(f"Mask: {args.mask}")
  try:
    image = load_image(args.image)
    mask = load_mask(args.mask)
  except FileNotFoundError as e:
    print(f"Error: Could not find input file. {e}", file=sys.stderr)
    sys.exit(1)

  print("[2/4] Initializing rendering pipeline...")
  segmenter = ManualMaskSegmentor(mask)
  estimator = AnalyticMotionEstimator(amplitude_px=args.amplitude)
  config = RenderConfig(n_frames=args.frames)
  
  svc = RenderService(segmenter, estimator, config)

  print(f"[3/4] Rendering {args.frames} frames (this may take a moment)...")
  start_time = time.time()
  cine = svc.render(image)
  elapsed = time.time() - start_time
  print(f"Render complete in {elapsed:.2f} seconds.")

  print(f"[4/4] Exporting deliverable to {args.out}...")
  save_gif(cine, args.out)
  
  print("Done! Cinemagraph successfully generated.")


if __name__ == "__main__":
  main()