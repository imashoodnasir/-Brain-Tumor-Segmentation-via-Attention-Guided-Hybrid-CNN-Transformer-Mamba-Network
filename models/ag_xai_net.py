from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.decoder import AttentionGuidedDecoder
from models.encoder import HybridEncoder
from models.explainability import ExplainabilityModule


class AGXAINet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        use_transformer: bool = True,
        use_state_space: bool = True,
        use_spatial_attention: bool = True,
        use_channel_attention: bool = True,
        use_cross_scale_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = HybridEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
            use_transformer=use_transformer,
            use_state_space=use_state_space,
        )
        self.decoder = AttentionGuidedDecoder(
            base_channels=base_channels,
            use_spatial_attention=use_spatial_attention,
            use_channel_attention=use_channel_attention,
            use_cross_scale_fusion=use_cross_scale_fusion,
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
        )
        self.explainer = ExplainabilityModule()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        self.explainer.register(feats["f4"])
        decoded, attn_maps = self.decoder(feats)
        logits = self.segmentation_head(decoded)
        probs = torch.sigmoid(logits)
        decoder_attention = self.explainer.aggregate_decoder_attention(attn_maps, logits.shape[-2:])
        return {
            "logits": logits,
            "probs": probs,
            "decoder_attention": decoder_attention,
            "attn_maps": attn_maps,
            "features": feats,
        }
