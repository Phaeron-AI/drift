from __future__ import annotations

import numpy as np

from ..types import Image, Mask, Region

class ManualMaskSegmentor:
  def __init__(self, subject_mask: Mask)-> None:
    self._subject = subject_mask
  
  def segment(self, image: Image)-> list[Region]:
    subject = (self._subject > 0.5).astype(np.float64)
    background = 1.0-subject
    return [
      Region(name="subject", mask=subject, is_locked=True, strength=0.0),
      Region(name="background", mask=background, strength=1.0),
    ]
  

class SAMSegmenter:
  def __init__(self, checkpoint: str, device: str = "cuda")-> None:
    self.device = device

    try:
      from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
      raise ImportError(
        "The 'segment_anything' package is required. "
        "Install via: pip install git+https://github.com/facebookresearch/segment-anything.git"
      )

    model_type = "vit_h"
    if "vit_l" in checkpoint:
      model_type = "vit_l"
    elif "vit_b" in checkpoint:
      model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=self.device)
    self.mask_generator = SamAutomaticMaskGenerator(sam)


  def segment(self, image: Image)-> list[Region]:
    if image.dtype in [np.float32, np.float64]:
      img_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
      img_uint8 = image
    
    h, w = img_uint8.shape[:2]
    masks = self.mask_generator.generate(img_uint8)

    if not masks:
      return [Region(name="background", mask=np.ones((h, w), dtype=np.float64), strength=1.0)]

    # HINT 2: Largest central person-like mask is the subject
    center_x, center_y = w / 2.0, h / 2.0
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    best_score = -1.0
    subject_idx = -1

    for i, ann in enumerate(masks):
      x, y, bw, bh = ann['bbox']
      mask_center_x = x + bw / 2.0
      mask_center_y = y + bh / 2.0
      
      dist = np.sqrt((mask_center_x - center_x)**2 + (mask_center_y - center_y)**2)
      normalized_dist = dist / max_dist

      score = ann['area'] / (1.0 + normalized_dist * 5.0)
      
      if score > best_score:
        best_score = score
        subject_idx = i
    
    subject_mask = masks[subject_idx]['segmentation'].astype(np.float64)
    regions = [
      Region(name="subject", mask=subject_mask, is_locked=True, strength=0.0)
    ]

    remaining_masks = [m for i, m in enumerate(masks) if i != subject_idx]
    remaining_masks.sort(key=lambda x: x['area'], reverse=True)

    occupied = subject_mask.copy()

    count = 1
    for ann in remaining_masks:
      raw_masks = ann["segmentation"].astype(np.float64)

      exclusive_mask = np.clip(raw_masks - occupied, 0.0, 1.0)

      if exclusive_mask.sum() > (h * w * 0.01):
        regions.append(Region(name=f"background_region_{count}", mask=exclusive_mask, strength=1.0))
        occupied = np.clip(occupied + exclusive_mask, 0.0, 1.0)
        count += 1
    
    leftover_mask = np.clip(1.0 - occupied, 0.0, 1.0)
    if leftover_mask.sum() > 0:
      regions.append(Region(name="background_leftover", mask=leftover_mask, strength=1.0))
    
    return regions