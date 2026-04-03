from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        out = self.act(out)
        return x + out


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            ResidualDepthwiseBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        b, n, c = x.shape
        device = x.device
        position = torch.arange(n, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, c, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / c))
        pe = torch.zeros(n, c, device=device)
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)


class TransformerBottleneck(nn.Module):
    def __init__(self, dim: int, heads: int = 8, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pos_enc = PositionalEncoding2D(dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        tokens = self.pos_enc(tokens)
        tokens = self.encoder(tokens)
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        return out


class StateSpace2D(nn.Module):
    """
    Self-contained Mamba-inspired selective state-space module.
    Sequence is scanned horizontally and vertically using learned gating.
    This avoids external dependencies while preserving the intended design.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(dim, dim, kernel_size=1)
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def _scan_width(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out = []
        state = torch.zeros((b, c, h), device=x.device, dtype=x.dtype)
        for i in range(w):
            token = x[:, :, :, i]
            state = 0.9 * state + token
            out.append(state.unsqueeze(-1))
        return torch.cat(out, dim=-1)

    def _scan_height(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out = []
        state = torch.zeros((b, c, w), device=x.device, dtype=x.dtype)
        for i in range(h):
            token = x[:, :, i, :]
            state = 0.9 * state + token
            out.append(state.unsqueeze(-2))
        return torch.cat(out, dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.in_proj(x)
        gate = torch.sigmoid(self.gate(x_proj))
        local = self.dw(x_proj)
        scan_w = self._scan_width(x_proj)
        scan_h = self._scan_height(x_proj)
        fused = gate * (local + scan_w + scan_h) / 3.0 + (1.0 - gate) * x_proj
        fused = self.out_proj(fused)
        fused = self.norm(fused)
        return x + fused


class SpatialAxialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 8, 8)
        self.q = nn.Conv2d(channels, reduced, 1)
        self.k = nn.Conv2d(channels, reduced, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Row attention
        q_row = q.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        k_row = k.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        a_row = torch.softmax(torch.bmm(q_row, k_row), dim=-1)
        v_row = v.permute(0, 2, 3, 1).reshape(b * h, w, c)
        out_row = torch.bmm(a_row, v_row).reshape(b, h, w, c).permute(0, 3, 1, 2)

        # Column attention
        q_col = q.permute(0, 3, 2, 1).reshape(b * w, h, -1)
        k_col = k.permute(0, 3, 1, 2).reshape(b * w, -1, h)
        a_col = torch.softmax(torch.bmm(q_col, k_col), dim=-1)
        v_col = v.permute(0, 3, 2, 1).reshape(b * w, h, c)
        out_col = torch.bmm(a_col, v_col).reshape(b, w, h, c).permute(0, 3, 2, 1)

        attn_map = (out_row.mean(dim=1, keepdim=True) + out_col.mean(dim=1, keepdim=True)) / 2.0
        out = x + self.gamma * (out_row + out_col) / 2.0
        return out, attn_map


class ChannelSEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, _, _ = x.shape
        pooled = x.mean(dim=(2, 3))
        weights = torch.sigmoid(self.fc2(F.gelu(self.fc1(pooled)))).view(b, c, 1, 1)
        out = x * weights
        attn_map = out.mean(dim=1, keepdim=True)
        return out, attn_map
