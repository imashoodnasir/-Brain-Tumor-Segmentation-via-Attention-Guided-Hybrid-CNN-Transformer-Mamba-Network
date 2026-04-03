from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.brain_tumor_dataset import BrainTumorSegmentationDataset
from engine.trainer import Trainer
from models.ag_xai_net import AGXAINet
from utils.transforms import build_train_transforms, build_val_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict):
    ds_cfg = cfg["dataset"]

    train_ds = BrainTumorSegmentationDataset(
        root=ds_cfg["train_root"] if ds_cfg["train_csv"] is None else None,
        csv_file=ds_cfg["train_csv"],
        image_dirname=ds_cfg["image_dirname"],
        mask_dirname=ds_cfg["mask_dirname"],
        image_size=ds_cfg["image_size"],
        transforms=build_train_transforms(cfg["augmentation"]["train"]),
        preprocess_cfg=ds_cfg["preprocess"],
    )
    val_ds = BrainTumorSegmentationDataset(
        root=ds_cfg["val_root"] if ds_cfg["val_csv"] is None else None,
        csv_file=ds_cfg["val_csv"],
        image_dirname=ds_cfg["image_dirname"],
        mask_dirname=ds_cfg["mask_dirname"],
        image_size=ds_cfg["image_size"],
        transforms=build_val_transforms(cfg["augmentation"].get("val")),
        preprocess_cfg=ds_cfg["preprocess"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=ds_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=ds_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)

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

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    metrics = trainer.fit()
    print("Final validation metrics:", metrics)


if __name__ == "__main__":
    main()
