from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.brain_tumor_dataset import BrainTumorSegmentationDataset
from engine.evaluator import evaluate_model
from models.ag_xai_net import AGXAINet
from utils.transforms import build_val_transforms


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = cfg["dataset"]
    test_ds = BrainTumorSegmentationDataset(
        root=ds_cfg["test_root"] if ds_cfg["test_csv"] is None else None,
        csv_file=ds_cfg["test_csv"],
        image_dirname=ds_cfg["image_dirname"],
        mask_dirname=ds_cfg["mask_dirname"],
        image_size=ds_cfg["image_size"],
        transforms=build_val_transforms(cfg["augmentation"].get("val")),
        preprocess_cfg=ds_cfg["preprocess"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=ds_cfg["num_workers"],
        pin_memory=True,
    )

    model = AGXAINet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        transformer_dim=cfg["model"]["transformer_dim"],
        transformer_heads=cfg["model"]["transformer_heads"],
        transformer_layers=cfg["model"]["transformer_layers"],
        dropout=cfg["model"]["dropout"],
        use_transformer=cfg["model"]["use_transformer"],
        use_state_space=cfg["model"]["use_state_space"],
        use_spatial_attention=cfg["model"]["use_spatial_attention"],
        use_channel_attention=cfg["model"]["use_channel_attention"],
        use_cross_scale_fusion=cfg["model"]["use_cross_scale_fusion"],
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        threshold=cfg["eval"]["threshold"],
        save_predictions=cfg["eval"].get("save_predictions", False),
        output_dir=cfg["output_dir"],
    )
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
