from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    heatmap: Optional[np.ndarray],
    out_path: str | Path,
    alpha: float = 0.35,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = image.squeeze()
    mask = mask.squeeze()
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2RGB)

    mask_rgb = np.zeros_like(image_rgb)
    mask_rgb[..., 0] = (mask > 0).astype(np.uint8) * 255
    overlay = cv2.addWeighted(image_rgb, 1.0, mask_rgb, alpha, 0)

    if heatmap is not None:
        heatmap = heatmap.squeeze()
        heatmap_u8 = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.75, heatmap_color, 0.25, 0)

    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
