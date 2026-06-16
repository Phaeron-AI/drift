from __future__ import annotations

import torch
from ..losses import l1_l2_reconstruction_loss

def main()-> None:
  torch.manual_seed(0)
  B, C, h, w = 2, 4, 16, 16
  H = W = 128
 
  target = torch.randn(B, C, h, w)

  mask = torch.zeros(B, 1, H, W)
  mask[:, :, : H // 2, :] = 1.0

  perfect = l1_l2_reconstruction_loss(target.clone(), target, mask, mask_weight=4.0)
  assert perfect.item() < 1e-5, f"perfect prediction should be ~0, got {perfect.item()}"

  wrong = l1_l2_reconstruction_loss(torch.zeros_like(target), target, mask, mask_weight=4.0)
  assert wrong.item() > 0, f"wrong prediction should be > 0, got {wrong.item()}"

  err_inside = target.clone()
  err_inside[:, :, : h // 2, :] += 5.0          # big error in the masked (top) region
  err_outside = target.clone()
  err_outside[:, :, h // 2 :, :] += 5.0         # same error in the unmasked (bottom) region
  loss_inside = l1_l2_reconstruction_loss(err_inside, target, mask, mask_weight=4.0)
  loss_outside = l1_l2_reconstruction_loss(err_outside, target, mask, mask_weight=4.0)
  assert loss_inside.item() > loss_outside.item(), (
    f"error inside mask should cost MORE: inside={loss_inside.item():.4f} "
    f"outside={loss_outside.item():.4f}"
  )
 
  # 4) lambda monotonicity: with error inside the mask, larger weight -> larger loss
  low = l1_l2_reconstruction_loss(err_inside, target, mask, mask_weight=1.0)
  high = l1_l2_reconstruction_loss(err_inside, target, mask, mask_weight=8.0)
  assert high.item() > low.item(), (
    f"bigger mask_weight should raise loss for in-mask error: "
    f"low={low.item():.4f} high={high.item():.4f}"
  )
 
  print("reconstruction loss OK:")
  print(f"  perfect={perfect.item():.2e}  wrong={wrong.item():.4f}")
  print(f"  inside={loss_inside.item():.4f} > outside={loss_outside.item():.4f}")
  print(f"  lambda: low={low.item():.4f} < high={high.item():.4f}")

if __name__ == "__main__":
  main()