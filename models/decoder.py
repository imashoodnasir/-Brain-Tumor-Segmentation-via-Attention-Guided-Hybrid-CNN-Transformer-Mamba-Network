from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ChannelSEAttention, ConvNormAct, SpatialAxialAttention
from models.fusion import GatedCrossScaleFusion


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_spatial_attention: bool = True,
        use_channel_attention: bool = True,
        use_cross_scale_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention
        self.use_cross_scale_fusion = use_cross_scale_fusion

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.pre_fuse = ConvNormAct(in_ch + skip_ch, out_ch, 3, 1)

        self.spatial_attn = SpatialAxialAttention(out_ch)
        self.channel_attn = ChannelSEAttention(out_ch)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.cross_scale = GatedCrossScaleFusion(out_ch)
        self.refine = nn.Sequential(
            ConvNormAct(out_ch, out_ch, 3, 1),
            ConvNormAct(out_ch, out_ch, 3, 1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.pre_fuse(x)

        attention_maps = {}

        spatial_out = x
        spatial_map = x.mean(dim=1, keepdim=True)
        if self.use_spatial_attention:
            spatial_out, spatial_map = self.spatial_attn(x)

        channel_out = x
        channel_map = x.mean(dim=1, keepdim=True)
        if self.use_channel_attention:
            channel_out, channel_map = self.channel_attn(x)

        x = self.alpha * spatial_out + (1.0 - self.alpha) * channel_out
        attention_maps["spatial"] = spatial_map
        attention_maps["channel"] = channel_map

        if self.use_cross_scale_fusion:
            x, cross_map = self.cross_scale(x, skip)
        else:
            cross_map = x.mean(dim=1, keepdim=True)

        attention_maps["cross_scale"] = cross_map
        x = self.refine(x)
        attention_maps["decoder"] = x.mean(dim=1, keepdim=True)
        return x, attention_maps


class AttentionGuidedDecoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        use_spatial_attention: bool = True,
        use_channel_attention: bool = True,
        use_cross_scale_fusion: bool = True,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stage3 = DecoderStage(c4, c3, c3, use_spatial_attention, use_channel_attention, use_cross_scale_fusion)
        self.stage2 = DecoderStage(c3, c2, c2, use_spatial_attention, use_channel_attention, use_cross_scale_fusion)
        self.stage1 = DecoderStage(c2, c1, c1, use_spatial_attention, use_channel_attention, use_cross_scale_fusion)

    def forward(self, feats: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_dict: Dict[str, torch.Tensor] = {}
        x, attn3 = self.stage3(feats["f4"], feats["f3"])
        x, attn2 = self.stage2(x, feats["f2"])
        x, attn1 = self.stage1(x, feats["f1"])

        for prefix, attn in zip(["s3", "s2", "s1"], [attn3, attn2, attn1]):
            for k, v in attn.items():
                attn_dict[f"{prefix}_{k}"] = v

        return x, attn_dict
