from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    scaler = _grad_scaler(amp and device.type == "cuda")
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_cnn(model: nn.Module, loader, criterion, device: torch.device, amp: bool, desc: str = "eval") -> dict:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    probabilities: list[np.ndarray] = []

    for images, labels, batch_paths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * images.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        probabilities.extend(probs.detach().cpu().numpy())
        paths.extend(list(batch_paths))

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "probabilities": np.asarray(probabilities),
        "paths": paths,
    }


def history_row(epoch: int, train_metrics: dict, val_metrics: dict, lr: float) -> dict:
    row = defaultdict(float)
    row["epoch"] = epoch
    row["lr"] = lr
    for key in ("loss", "accuracy", "f1_macro"):
        row[f"train_{key}"] = train_metrics[key]
        row[f"val_{key}"] = val_metrics[key]
    return dict(row)
