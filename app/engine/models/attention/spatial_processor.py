from __future__ import annotations
# pyright: reportAttributeAccessIssue=false

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusers.models.attention_processor import AttnProcessor2_0

from .trajectory_encoder import TrajectoryEncoder
from .mask import MaskToBias


class SpatialProcessor:
  def __init__(self, traj_encoder: TrajectoryEncoder, mask_to_bias: MaskToBias, n_traj: int = 1) -> None:
    self.traj_encoder = traj_encoder
    self.mask_to_bias = mask_to_bias
    self.n_traj = n_traj
    self.fallback_attn = AttnProcessor2_0()

  def __call__(
    self,
    attn,
    hidden_states: Tensor,
    encoder_hidden_states: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    temb: Optional[Tensor] = None,
    trajectory: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    **kwargs,
  ) -> Tensor:
    if trajectory is None or mask is None:
      return self.fallback_attn(
        attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kwargs
      )

    residual = hidden_states

    if attn.spatial_norm is not None:
      hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim          # FIX 2: .ndim, not .n_dim
    if input_ndim == 4:
      batch_size, channel, height, width = hidden_states.shape
      hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size = hidden_states.shape[0]

    if attention_mask is not None:
      attention_mask = attn.prepare_attention_mask(attention_mask, hidden_states.shape[1], batch_size)
      attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])  # type: ignore

    if attn.group_norm is not None:
      hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    # --- build the conditioning context (Option 1: concat BEFORE projection) ---
    traj_token = self.traj_encoder(trajectory)                  # [B, n_traj, E]
    if encoder_hidden_states is not None:
      # Mirror AttnProcessor2_0's optional cross-norm on the TEXT only; leave traj as-is.
      if attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
      context = torch.cat([encoder_hidden_states, traj_token], dim=1)  # [B, N_text+n_traj, E]  # type: ignore
      n_text_eff = encoder_hidden_states.shape[1] # type: ignore
    else:
      context = traj_token                                      # [B, n_traj, E]
      n_text_eff = 0

    q = attn.to_q(hidden_states)
    k = attn.to_k(context)
    v = attn.to_v(context)

    heads = attn.heads
    head_dim = q.shape[-1] // heads
    q = q.view(batch_size, -1, heads, head_dim).transpose(1, 2)   # [B, heads, S_q,  head_dim]
    k = k.view(batch_size, -1, heads, head_dim).transpose(1, 2)   # [B, heads, S_kv, head_dim]
    v = v.view(batch_size, -1, heads, head_dim).transpose(1, 2)

    # --- mask -> additive bias, sized to this block; injected inside the softmax ---
    bias = self.mask_to_bias(mask, n_text_eff).to(q.dtype)       # [B, heads, S_q, S_kv]

    out = F.scaled_dot_product_attention(
      q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
    )                                                            # [B, heads, S_q, head_dim]

    out = out.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
    out = out.to(q.dtype)

    out = attn.to_out[0](out)    # Linear (LoRA-wrapped)
    out = attn.to_out[1](out)    # Dropout

    if input_ndim == 4:
      out = out.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
      out = out + residual
    out = out / attn.rescale_output_factor

    return out

def install_spatial_processors(
  unet: nn.Module,
  traj_encoder: TrajectoryEncoder,
  target_latent_sizes: tuple[int, ...] = (8, 16, 32),   # FIX 1: builtin tuple, no typing.Tuple
  n_traj: int = 1,
) -> nn.ModuleDict:
  if hasattr(unet, "attn_processors") and hasattr(unet, "set_attn_processor"):
    proc_host = unet
  elif hasattr(unet, "base_model"):
    proc_host = unet.base_model.model
  else:
    raise AttributeError(
      "Could not find attn_processors on the U-Net or its PEFT base. "
      f"Got type {type(unet)}; inspect it and point proc_host at the real UNet2DConditionModel."
    )

  latent_sizes: dict[str, int] = {}
  hooks = []

  def make_hook(name):
    def hook(module, args, kwargs):
      hs = args[0] if args else kwargs.get("hidden_states")
      latent_sizes[name] = int(round(math.sqrt(hs.shape[1])))   # [B, S_q, C] -> sqrt(S_q)
    return hook

  cross_attn_names = [
    n[: -len(".processor")]
    for n in proc_host.attn_processors.keys() # type: ignore
    if n.endswith("attn2.processor")
  ]

  for name in cross_attn_names:
    module = proc_host.get_submodule(name)
    hooks.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))

  target_model = unet.base_model.model if hasattr(unet, "base_model") else unet
  try:
    first_param = next(target_model.parameters())
    dev, dt = first_param.device, first_param.dtype
  except StopIteration:
    dev, dt = torch.device("cpu"), torch.float32

  in_ch = proc_host.conv_in.in_channels
  z = torch.zeros(1, in_ch, 64, 64, device=dev, dtype=dt)        # standard SD1.5 latent  # type: ignore
  t = torch.zeros(1, device=dev, dtype=torch.long)
  ctx = torch.zeros(1, 77, 768, device=dev, dtype=dt)        # dummy text context
  with torch.no_grad():
    target_model(z, t, encoder_hidden_states=ctx) # type: ignore

  for h in hooks:
    h.remove()

  new_procs = dict(proc_host.attn_processors)   # start from current -> untouched blocks keep theirs  # type: ignore
  mask_modules = nn.ModuleDict()

  for name in cross_attn_names:
    ls = latent_sizes.get(name)
    if ls is None:
      print(f"Warning: could not determine latent size for {name}; skipping.")
      continue
    if ls not in target_latent_sizes:
      continue                                  # e.g. skip the 64x64 block

    attn = proc_host.get_submodule(name)
    mtb = MaskToBias(latent_size=ls, n_heads=attn.heads, n_traj=n_traj)  # type: ignore
    new_procs[name + ".processor"] = SpatialProcessor(traj_encoder, mtb, n_traj=n_traj) # type: ignore
    mask_modules[name.replace(".", "_")] = mtb  # ModuleDict keys can't contain '.'

  proc_host.set_attn_processor(new_procs) # type: ignore
  return mask_modules