from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class ExplainabilityModule:
    def __init__(self):
        self.feature_map: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None

    def register(self, tensor: torch.Tensor) -> None:
        self.feature_map = tensor

        def _hook(grad: torch.Tensor) -> None:
            self.gradient = grad

        tensor.register_hook(_hook)

    def grad_cam(self, logits: torch.Tensor) -> torch.Tensor:
        if self.feature_map is None or self.gradient is None:
            raise RuntimeError("Call backward() before requesting Grad-CAM.")
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize(cam)
        return cam

    @staticmethod
    def aggregate_decoder_attention(attn_maps: Dict[str, torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        maps = []
        for _, v in attn_maps.items():
            maps.append(F.interpolate(v, size=output_size, mode="bilinear", align_corners=False))
        stacked = torch.stack(maps, dim=0).mean(dim=0)
        return ExplainabilityModule._normalize(stacked)

    @staticmethod
    def combine(grad_cam_map: torch.Tensor, decoder_map: torch.Tensor, lam: float = 0.5) -> torch.Tensor:
        return ExplainabilityModule._normalize(lam * grad_cam_map + (1.0 - lam) * decoder_map)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        xmin = x.amin(dim=(2, 3), keepdim=True)
        xmax = x.amax(dim=(2, 3), keepdim=True)
        return (x - xmin) / (xmax - xmin + 1e-8)
