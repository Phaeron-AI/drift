from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ..config import HybridConfig
from ..models.hybrid import HybridSpatialDiffusion

logger = logging.getLogger("drift.sweep")


def _load_source(path: str, cfg: HybridConfig) -> torch.Tensor:
  img = Image.open(path).convert("RGB")
  t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
  return F.interpolate(t.unsqueeze(0), size=(cfg.img_size, cfg.img_size),
                       mode="bilinear", align_corners=False)


def _step_of(p: Path) -> int:
  m = re.fullmatch(r"step_(\d+)\.pt", p.name)
  return int(m.group(1)) if m else -1


def _conv_in_source_norm(state: dict) -> float:
  w = [v for k, v in state.items() if "conv_in" in k and v.ndim == 4]
  return round(w[0][:, 4:].float().norm().item(), 4) if w else float("nan")


@torch.no_grad()
def sweep(cfg: HybridConfig, args) -> None:
  device = cfg.device

  # collect checkpoints to compare (all step_*.pt in the dir, or a subset via --steps-list)
  ckpts = sorted([p for p in args.ckpt_dir.glob("step_*.pt") if _step_of(p) >= 0],
                 key=_step_of)
  if args.steps_list:
    keep = {int(s) for s in args.steps_list.split(",")}
    ckpts = [p for p in ckpts if _step_of(p) in keep]
  if not ckpts:
    raise SystemExit(f"no checkpoints found in {args.ckpt_dir}")
  logger.info(f"sweeping {len(ckpts)} checkpoints: {[_step_of(p) for p in ckpts]}")

  # build the model ONCE; reload weights per checkpoint (cheap vs rebuilding the U-Net each time)
  model = HybridSpatialDiffusion(cfg)
  model.eval()

  # source + conditioning are identical across all checkpoints -> fair comparison
  src01 = _load_source(args.source, cfg).to(device)
  src_norm = (src01 * 2.0 - 1.0).to(cfg.compute_dtype)
  source_latent = model.encode(src_norm)
  grab = torch.tensor([[args.grab_x, args.grab_y]], device=device, dtype=torch.float32)
  mask = model.predict_mask(src01, grab).to(cfg.compute_dtype)
  traj = torch.tensor([[args.dx / cfg.img_size, args.dy / cfg.img_size]],
                      device=device, dtype=cfg.compute_dtype)

  # FIX a seed so the only thing that varies between panels is the CHECKPOINT, not the noise
  panels = []
  src_pil = Image.fromarray((src01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

  for ckpt in ckpts:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict(state, strict=False)
    norm = _conv_in_source_norm(state)

    torch.manual_seed(args.seed)                 # same init noise for every checkpoint
    scheduler = model.backbone.scheduler
    scheduler.set_timesteps(args.steps)
    h = cfg.latent_size
    x = torch.randn(1, cfg.latent_channels, h, h, device=device, dtype=cfg.compute_dtype)
    x = x * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
      with torch.autocast(device_type="cuda", dtype=cfg.compute_dtype):
        eps = model(x, t.to(device).unsqueeze(0), traj, mask, source_latent, text_embeds=None)
      x = scheduler.step(eps, t, x).prev_sample # type: ignore

    img = ((model.decode(x).float().clamp(-1, 1) + 1.0) / 2.0)
    gen = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    panels.append((_step_of(ckpt), norm, Image.fromarray(gen)))
    logger.info(f"step {_step_of(ckpt)}: source-norm {norm}")

  # lay out: source on the left, then one generated panel per checkpoint, left->right
  W, H = src_pil.size
  cols = 1 + len(panels)
  canvas = Image.new("RGB", (W * cols + 10 * (cols - 1), H + 24), (15, 15, 20))
  canvas.paste(src_pil, (0, 24))
  d = ImageDraw.Draw(canvas)
  d.text((4, 6), f"SOURCE  (dx={args.dx}, dy={args.dy})", fill=(200, 200, 200))
  sx, sy, ex, ey = args.grab_x, args.grab_y, args.grab_x + args.dx, args.grab_y + args.dy
  d.ellipse([sx-4, sy+24-4, sx+4, sy+24+4], outline=(0, 255, 0), width=2)
  d.line([sx, sy+24, ex, ey+24], fill=(255, 255, 0), width=2)
  for i, (step, norm, pil) in enumerate(panels):
    x0 = (W + 10) * (i + 1)
    canvas.paste(pil, (x0, 24))
    d.text((x0 + 4, 6), f"step {step}  src-norm {norm}", fill=(200, 200, 200))

  out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(out)
  logger.info(f"saved sweep -> {out}  ({len(panels)} checkpoints, fixed seed {args.seed})")


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
  p = argparse.ArgumentParser(description="Sample one source+trajectory across many checkpoints.")
  p.add_argument("--source", type=str, required=True)
  p.add_argument("--ckpt-dir", type=Path, required=True, help="dir containing step_*.pt")
  p.add_argument("--steps-list", type=str, default=None,
                 help="comma-separated steps to include, e.g. 4000,6000,8000 (default: all)")
  p.add_argument("--grab-x", type=float, required=True)
  p.add_argument("--grab-y", type=float, required=True)
  p.add_argument("--dx", type=float, required=True)
  p.add_argument("--dy", type=float, required=True)
  p.add_argument("--steps", type=int, default=50, help="diffusion steps")
  p.add_argument("--seed", type=int, default=0, help="fixed init-noise seed (fair comparison)")
  p.add_argument("--out", type=str, default="viz/sweep.png")
  args = p.parse_args()
  cfg = HybridConfig()
  sweep(cfg, args)


if __name__ == "__main__":
  main()