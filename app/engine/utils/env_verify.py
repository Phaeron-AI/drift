import torch
import logging

from ..config.schema import HybridConfig

logger = logging.getLogger("drift.phase1")

def verify_environment(cfg: HybridConfig) -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable — check your torch install / driver.")

  cap = torch.cuda.get_device_capability(0)
  if cap != (12, 0):
    raise RuntimeError(f"Expected sm_120 (12, 0); got {cap}.")

  if "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError(
      f"torch built without Blackwell kernels. arch_list={torch.cuda.get_arch_list()}. "
      "Reinstall from the cu128/cu130 index."
    )

  if not torch.cuda.is_bf16_supported():
    raise RuntimeError("bfloat16 not supported on this device.")

  from torch.nn.attention import SDPBackend, sdpa_kernel
  q = torch.randn(1, 8, 128, 64, device=cfg.device, dtype=cfg.compute_dtype)
  try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
      torch.nn.functional.scaled_dot_product_attention(q, q, q)
    logger.info("SDPA flash backend OK")
  except RuntimeError:
    logger.warning("Flash SDPA unavailable — falling back to math/efficient kernel.")

  free_b, total_b = torch.cuda.mem_get_info()
  logger.info(f"GPU free:  {free_b  / 1024**3:.2f} GiB")
  logger.info(f"GPU total: {total_b / 1024**3:.2f} GiB")
  logger.info(f"Device: {torch.cuda.get_device_name(0)}  capability={cap}")

if __name__ == "__main__":
  config = HybridConfig()
  verify_environment(config)