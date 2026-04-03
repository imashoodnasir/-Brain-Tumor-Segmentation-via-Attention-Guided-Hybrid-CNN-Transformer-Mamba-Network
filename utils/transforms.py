from __future__ import annotations

from typing import Dict

import albumentations as A


def build_train_transforms(cfg: Dict):
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.get("horizontal_flip", 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=0.10,
                rotate_limit=10,
                border_mode=0,
                p=cfg.get("shift_scale_rotate", 0.5),
            ),
            A.ElasticTransform(alpha=30, sigma=5, alpha_affine=5, p=cfg.get("elastic", 0.2)),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=cfg.get("brightness_contrast", 0.2),
            ),
            A.GaussNoise(var_limit=(5.0, 25.0), p=cfg.get("gauss_noise", 0.1)),
        ]
    )


def build_val_transforms(cfg: Dict | None = None):
    return A.Compose([])
