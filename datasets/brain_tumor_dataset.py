from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.preprocess import preprocess_image_and_mask


@dataclass
class SampleRecord:
    image_path: str
    mask_path: str
    dataset: str = "unknown"
    label: int = 1


def _scan_root(root: Path, image_dirname: str, mask_dirname: str) -> List[SampleRecord]:
    image_dir = root / image_dirname
    mask_dir = root / mask_dirname
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Could not find image/mask directories under: {root}")

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])
    records: List[SampleRecord] = []
    for img_path in image_files:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            records.append(
                SampleRecord(
                    image_path=str(img_path),
                    mask_path=str(mask_path),
                    dataset=root.name,
                    label=1,
                )
            )
    if not records:
        raise RuntimeError(f"No matched image/mask pairs found in: {root}")
    return records


class BrainTumorSegmentationDataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        csv_file: Optional[str] = None,
        image_dirname: str = "images",
        mask_dirname: str = "masks",
        image_size: int = 256,
        transforms: Optional[Callable] = None,
        preprocess_cfg: Optional[Dict] = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.csv_file = Path(csv_file) if csv_file is not None else None
        self.image_dirname = image_dirname
        self.mask_dirname = mask_dirname
        self.image_size = image_size
        self.transforms = transforms
        self.preprocess_cfg = preprocess_cfg or {}

        if self.csv_file is not None:
            self.records = self._load_csv(self.csv_file)
        elif self.root is not None:
            self.records = _scan_root(self.root, image_dirname, mask_dirname)
        else:
            raise ValueError("Either root or csv_file must be provided.")

    @staticmethod
    def _load_csv(csv_path: Path) -> List[SampleRecord]:
        df = pd.read_csv(csv_path)
        required = {"image_path", "mask_path"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain columns: {required}")
        records: List[SampleRecord] = []
        for row in df.to_dict(orient="records"):
            records.append(
                SampleRecord(
                    image_path=row["image_path"],
                    mask_path=row["mask_path"],
                    dataset=row.get("dataset", "unknown"),
                    label=int(row.get("label", 1)),
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_grayscale(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = self._read_grayscale(record.image_path)
        mask = self._read_grayscale(record.mask_path)

        image, mask, brain_mask = preprocess_image_and_mask(
            image=image,
            mask=mask,
            image_size=self.image_size,
            skull_strip=self.preprocess_cfg.get("skull_strip", True),
            n4_bias_correction=self.preprocess_cfg.get("n4_bias_correction", True),
            zscore=self.preprocess_cfg.get("zscore", True),
        )

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask.astype(np.uint8))
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32)
        mask = (mask > 0).astype(np.float32)
        brain_mask = (brain_mask > 0).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        brain_mask = np.expand_dims(brain_mask, axis=0)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "brain_mask": torch.from_numpy(brain_mask),
            "label": torch.tensor(record.label, dtype=torch.long),
            "dataset_name": record.dataset,
            "image_path": record.image_path,
            "mask_path": record.mask_path,
        }
