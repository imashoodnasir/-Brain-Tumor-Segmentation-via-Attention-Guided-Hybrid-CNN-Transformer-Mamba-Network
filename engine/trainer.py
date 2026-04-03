from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.evaluator import evaluate_model
from engine.metrics import compute_segmentation_metrics
from losses.segmentation_losses import CompositeSegmentationLoss
from utils.ema import ModelEMA


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        train_cfg = cfg["train"]
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=max(1, train_cfg["warmup_epochs"])
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, train_cfg["epochs"] - train_cfg["warmup_epochs"]),
            eta_min=train_cfg["min_lr"],
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[train_cfg["warmup_epochs"]],
        )

        self.criterion = CompositeSegmentationLoss(cfg["loss"])
        self.scaler = GradScaler(enabled=train_cfg.get("amp", True))
        self.ema = ModelEMA(self.model, decay=train_cfg.get("ema_decay", 0.999))
        self.best_dice = -1.0
        self.patience = 0

        self.log_file = self.output_dir / "train_log.csv"
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_dice", "val_iou", "val_hd95", "val_assd"])

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "hd95": 0.0, "assd": 0.0}
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for batch in progress:
            images = batch["image"].to(self.device, dtype=torch.float32)
            masks = batch["mask"].to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg["train"].get("amp", True)):
                outputs = self.model(images)
                loss, loss_stats = self.criterion(
                    probs=outputs["probs"],
                    targets=masks,
                    attention_map=outputs["decoder_attention"],
                )

            self.scaler.scale(loss).backward()
            if self.cfg["train"].get("grad_clip", None):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)

            batch_metrics = compute_segmentation_metrics(
                outputs["probs"].detach(),
                masks.detach(),
                threshold=self.cfg["eval"]["threshold"],
            )

            total_loss += float(loss.detach().cpu())
            for k, v in batch_metrics.items():
                total_metrics[k] += v
            num_batches += 1

            progress.set_postfix({"loss": f"{loss_stats['total_loss']:.4f}", "dice": f"{batch_metrics['dice']:.4f}"})

        self.scheduler.step()

        avg_metrics = {k: v / max(1, num_batches) for k, v in total_metrics.items()}
        avg_metrics["loss"] = total_loss / max(1, num_batches)
        return avg_metrics

    def fit(self) -> Dict[str, float]:
        epochs = self.cfg["train"]["epochs"]
        early_stop_patience = self.cfg["train"]["early_stopping_patience"]

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(epoch)

            val_metrics = evaluate_model(
                self.ema.ema_model,
                self.val_loader,
                device=self.device,
                threshold=self.cfg["eval"]["threshold"],
                save_predictions=False,
            )

            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_metrics["loss"],
                    val_metrics["dice"],
                    val_metrics["iou"],
                    val_metrics["hd95"],
                    val_metrics["assd"],
                ])

            if val_metrics["dice"] > self.best_dice:
                self.best_dice = val_metrics["dice"]
                self.patience = 0
                torch.save(self.ema.ema_model.state_dict(), self.output_dir / "best_model.pt")
            else:
                self.patience += 1

            torch.save(self.ema.ema_model.state_dict(), self.output_dir / "last_model.pt")

            if self.patience >= early_stop_patience:
                break

        metrics = evaluate_model(
            self.ema.ema_model,
            self.val_loader,
            device=self.device,
            threshold=self.cfg["eval"]["threshold"],
            save_predictions=self.cfg["eval"].get("save_predictions", False),
            output_dir=self.output_dir,
        )
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
