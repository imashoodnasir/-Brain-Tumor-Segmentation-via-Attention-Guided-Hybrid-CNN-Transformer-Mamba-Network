from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from models.blocks import (
    ConvNormAct,
    DownsampleBlock,
    ResidualDepthwiseBlock,
    StateSpace2D,
    TransformerBottleneck,
)


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        use_transformer: bool = True,
        use_state_space: bool = True,
    ) -> None:
        super().__init__()
        chs = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.use_transformer = use_transformer
        self.use_state_space = use_state_space

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, chs[0], 3, 1),
            ResidualDepthwiseBlock(chs[0]),
        )
        self.stage2 = DownsampleBlock(chs[0], chs[1])
        self.stage3 = DownsampleBlock(chs[1], chs[2])
        self.stage4 = DownsampleBlock(chs[2], chs[3])

        self.to_transformer_dim = nn.Conv2d(chs[3], transformer_dim, kernel_size=1)
        self.from_transformer_dim = nn.Conv2d(transformer_dim, chs[3], kernel_size=1)
        self.transformer = TransformerBottleneck(transformer_dim, transformer_heads, transformer_layers, dropout)
        self.state_space = StateSpace2D(chs[3])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1 = self.stem(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        enhanced = f4
        if self.use_transformer:
            t = self.to_transformer_dim(f4)
            t = self.transformer(t)
            t = self.from_transformer_dim(t)
            enhanced = enhanced + t

        if self.use_state_space:
            enhanced = self.state_space(enhanced)

        return {"f1": f1, "f2": f2, "f3": f3, "f4": enhanced}
