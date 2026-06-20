from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import torch.nn.functional as F

from ..config import HybridConfig
from ..models.hybrid import HybridSpatialDiffusion

logger = logging.getLogger("drift.sample")


def _load_source(path: str, cfg: HybridConfig) -> torch.Tensor:
  # [1,3,size,size] in [0,1], squished to square -- SAME convention as training data.
  img = Image.open(path).convert("RGB")
  t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
  t = F.interpolate(t.unsqueeze(0), size=(cfg.img_size, cfg.img_size),
                    mode="bilinear", align_corners=False)
  return t


def _load_checkpoint(model: HybridSpatialDiffusion, ckpt_path: Path) -> None:
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
  missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
  if unexpected:
    logger.warning(f"{len(unexpected)} unexpected keys in checkpoint")
  logger.info(f"Loaded trainable weights from {ckpt_path} (step {ckpt.get('global_step', '?')})")


@torch.no_grad()
def sample(cfg: HybridConfig, args) -> None:
  device = cfg.device
  model = HybridSpatialDiffusion(cfg)
  _load_checkpoint(model, args.checkpoint)
  model.eval()

  # --- source appearance ---
  src01 = _load_source(args.source, cfg).to(device)          # [1,3,H,W] in [0,1]
  src_norm = (src01 * 2.0 - 1.0).to(cfg.compute_dtype)        # VAE wants [-1,1]
  source_latent = model.encode(src_norm)                     # [1,4,h,w] clean scaled latent

  # --- mask: SAM2 at the grab point (SAME path as precompute -> train/inference consistency) ---
  grab = torch.tensor([[args.grab_x, args.grab_y]], device=device, dtype=torch.float32)
  mask = model.predict_mask(src01, grab)                     # [1,1,H,W] in {0,1}
  mask = mask.to(cfg.compute_dtype)

  # --- trajectory: pixels -> normalized by img_size, EXACTLY as training computed it ---
  traj = torch.tensor([[args.dx / cfg.img_size, args.dy / cfg.img_size]],
                      device=device, dtype=cfg.compute_dtype)   # [1,2]
  logger.info(f"trajectory (normalized): {traj.tolist()}  | mask fg frac: {mask.mean().item():.3f}")

  # --- reverse diffusion ---
  scheduler = model.backbone.scheduler
  scheduler.set_timesteps(args.steps)
  h = cfg.latent_size
  x = torch.randn(1, cfg.latent_channels, h, h, device=device, dtype=cfg.compute_dtype)
  x = x * scheduler.init_noise_sigma

  for t in scheduler.timesteps:
    t_batch = t.to(device).unsqueeze(0)                      # [1]
    with torch.autocast(device_type="cuda", dtype=cfg.compute_dtype):
      eps = model(x, t_batch, traj, mask, source_latent, text_embeds=None)   # null-text = train
    x = scheduler.step(eps, t, x).prev_sample # type: ignore
    # NOTE[CFG]: no classifier-free guidance -- you did not train with conditioning dropout.
    #   To add it later: train with random null-conditioning, then here run cond+uncond and
    #   extrapolate eps = eps_uncond + w*(eps_cond - eps_uncond).

  # --- decode ---
  img = model.decode(x)                                      # [1,3,H,W] in [-1,1]-ish
  img = ((img.float().clamp(-1, 1) + 1.0) / 2.0)             # -> [0,1]
  gen = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

  # --- save source|generated with the trajectory arrow ---
  src_pil = Image.fromarray((src01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
  gen_pil = Image.fromarray(gen)
  W, H = src_pil.size
  canvas = Image.new("RGB", (W * 2 + 10, H), (0, 0, 0))
  canvas.paste(src_pil, (0, 0)); canvas.paste(gen_pil, (W + 10, 0))
  d = ImageDraw.Draw(canvas)
  sx, sy = args.grab_x, args.grab_y
  ex, ey = sx + args.dx, sy + args.dy
  d.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], outline=(0, 255, 0), width=2)     # grab on source
  d.line([sx, sy, ex, ey], fill=(255, 255, 0), width=2)                        # requested motion
  d.ellipse([W + 10 + ex - 4, ey - 4, W + 10 + ex + 4, ey + 4], outline=(255, 0, 0), width=2)
  out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(out)
  logger.info(f"saved {out}  (left: source + requested trajectory, right: generated)")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser(description="Generate a target frame from source + trajectory.")
  p.add_argument("--source", type=str, required=True, help="source image path")
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--grab-x", type=float, required=True, help="x of object grab point (img_size space)")
  p.add_argument("--grab-y", type=float, required=True, help="y of object grab point")
  p.add_argument("--dx", type=float, required=True, help="desired x displacement in pixels (img_size space)")
  p.add_argument("--dy", type=float, required=True, help="desired y displacement in pixels")
  p.add_argument("--steps", type=int, default=50, help="reverse diffusion steps")
  p.add_argument("--out", type=str, default="viz/sample.png")
  args = p.parse_args()
  cfg = HybridConfig()
  sample(cfg, args)


if __name__ == "__main__":
  main()