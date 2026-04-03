from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.metrics import compute_segmentation_metrics
from utils.visualization import save_overlay


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    save_predictions: bool = False,
    output_dir: str | None = None,
) -> Dict[str, float]:
    model.eval()
    aggregate = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "hd95": 0.0, "assd": 0.0}
    count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.float32)

        outputs = model(images)
        probs = outputs["probs"]

        metrics = compute_segmentation_metrics(probs, masks, threshold=threshold)
        for k, v in metrics.items():
            aggregate[k] += v
        count += 1

        if save_predictions and output_dir is not None and batch_idx < 20:
            out_root = Path(output_dir) / "predictions"
            for i in range(images.size(0)):
                image = images[i].detach().cpu().numpy()
                mask = probs[i].detach().cpu().numpy()
                heat = outputs["decoder_attention"][i].detach().cpu().numpy()
                img_path = Path(batch["image_path"][i]).stem
                save_overlay(image, mask, heat, out_root / f"{img_path}_overlay.png")

    if count == 0:
        return {k: 0.0 for k in aggregate}
    return {k: v / count for k, v in aggregate.items()}
