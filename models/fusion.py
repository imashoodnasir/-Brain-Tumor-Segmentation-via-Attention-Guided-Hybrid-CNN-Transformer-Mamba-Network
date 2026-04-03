from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossScaleFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 4, 8)
        self.q = nn.Conv2d(channels, reduced, 1, bias=False)
        self.k = nn.Conv2d(channels, reduced, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, decoder_feat: torch.Tensor, encoder_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = decoder_feat.shape
        q = self.q(decoder_feat).flatten(2).transpose(1, 2)  # [B, HW, C']
        k = self.k(encoder_feat).flatten(2)                  # [B, C', HW]
        attn = torch.softmax(torch.bmm(q, k), dim=-1)

        v = self.v(encoder_feat).flatten(2).transpose(1, 2)  # [B, HW, C]
        attended = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)

        gate = self.gate_mlp(torch.cat([decoder_feat, attended], dim=1))
        fused = gate * attended + (1.0 - gate) * decoder_feat
        fused = self.out(fused)
        attn_map = attended.mean(dim=1, keepdim=True)
        return fused, attn_map
