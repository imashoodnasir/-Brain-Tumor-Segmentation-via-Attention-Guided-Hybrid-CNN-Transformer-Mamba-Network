from __future__ import annotations

import copy
from typing import Iterator, Tuple

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for name, ema_v in self.ema_model.state_dict().items():
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(msd[name].detach(), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(msd[name])

    def state_dict(self):
        return self.ema_model.state_dict()
