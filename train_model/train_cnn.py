from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import deep_update, label_names, load_config, save_config
from src.data import make_cnn_loaders
from src.engine import evaluate_cnn, history_row, train_one_epoch
from src.metrics import save_classification_outputs
from src.models import build_model, get_feature_dim, normalize_model_name
from src.utils import ensure_dir, get_device, set_seed
from src.visualize import (
    plot_binary_curves,
    plot_class_distribution,
    plot_confusion,
    plot_misclassified_grid,
    plot_training_history,
)


try:
    from torch.utils.tensorboard import SummaryWriter as TensorBoardSummaryWriter
except Exception as exc:
    TensorBoardSummaryWriter = None
    TENSORBOARD_IMPORT_ERROR = exc
else:
    TENSORBOARD_IMPORT_ERROR = None


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a CNN backbone with Cross-Entropy.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_criterion(cfg: dict, train_loader, device: torch.device) -> nn.Module:
    if str(cfg["training"].get("class_weight", "none")).lower() != "balanced":
        return nn.CrossEntropyLoss()

    counts = train_loader.dataset.class_counts()
    num_classes = len(cfg["dataset"]["label_names"])
    total = sum(counts.values())
    weights = [total / (num_classes * max(counts.get(i, 1), 1)) for i in range(num_classes)]
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def save_checkpoint(path: Path, model, optimizer, epoch: int, score: float, cfg: dict, names: list[str]) -> None:
    model_name = normalize_model_name(str(cfg["model"].get("name", "resnet18")))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_f1_macro": score,
            "model_name": model_name,
            "feature_dim": get_feature_dim(model_name),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "label_names": names,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {
        "training": {
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "augment": False if args.no_augment else None,
        }
    }
    cfg = deep_update(cfg, overrides)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))
    output_dir = ensure_dir(cfg["training"]["output_dir"])
    save_config(cfg, output_dir / "config_used.yaml")

    names = label_names(cfg)
    train_loader, val_loader, test_loader = make_cnn_loaders(cfg)
    plot_class_distribution(
        {
            "train": train_loader.dataset.class_counts(),
            "val": val_loader.dataset.class_counts(),
            "test": test_loader.dataset.class_counts(),
        },
        names,
        output_dir / "class_distribution.png",
    )

    model_name = normalize_model_name(str(cfg["model"].get("name", "resnet18")))
    model = build_model(model_name, num_classes=len(names), pretrained=bool(cfg["model"].get("pretrained", True)))
    model.to(device)
    criterion = make_criterion(cfg, train_loader, device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    if TensorBoardSummaryWriter is None:
        print(f"TensorBoard is disabled because import failed: {TENSORBOARD_IMPORT_ERROR}")
        writer = NullSummaryWriter()
    else:
        writer = TensorBoardSummaryWriter(log_dir=str(output_dir / "tensorboard"))

    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"].get("patience", 0))
    amp = bool(cfg["training"].get("amp", True))

    print(f"Device: {device}")
    print(f"Backbone: {model_name}, feature_dim={get_feature_dim(model_name)}")
    print(f"Train/Val/Test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"Output: {output_dir}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, amp, epoch)
        val_metrics = evaluate_cnn(model, val_loader, criterion, device, amp, desc=f"val epoch {epoch}")
        current_lr = optimizer.param_groups[0]["lr"]
        row = history_row(epoch, train_metrics, val_metrics, current_lr)
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("f1_macro/train", train_metrics["f1_macro"], epoch)
        writer.add_scalar("f1_macro/val", val_metrics["f1_macro"], epoch)
        writer.add_scalar("lr", current_lr, epoch)

        scheduler.step(val_metrics["f1_macro"])
        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_f1={train_metrics['f1_macro']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        save_checkpoint(output_dir / "last_cnn.pt", model, optimizer, epoch, best_score, cfg, names)
        if val_metrics["f1_macro"] > best_score:
            best_score = float(val_metrics["f1_macro"])
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output_dir / "best_cnn.pt", model, optimizer, epoch, best_score, cfg, names)
        else:
            stale_epochs += 1

        plot_training_history(output_dir / "history.csv", output_dir / "training_curves.png")
        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best val F1={best_score:.4f}")
            break

    writer.close()

    best_state = torch.load(output_dir / "best_cnn.pt", map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    val_final = evaluate_cnn(model, val_loader, criterion, device, amp, desc="val best")
    test_final = evaluate_cnn(model, test_loader, criterion, device, amp, desc="test best")

    for split, result in [("val", val_final), ("test", test_final)]:
        save_classification_outputs(
            result["y_true"],
            result["y_pred"],
            names,
            output_dir,
            f"cnn_{split}",
            paths=result["paths"],
            probabilities=result["probabilities"],
        )
        plot_confusion(result["y_true"], result["y_pred"], names, output_dir / f"cnn_{split}_confusion.png", f"CNN {split}")
        plot_binary_curves(result["y_true"], result["probabilities"], output_dir, f"cnn_{split}", positive_label=1)
        plot_misclassified_grid(
            result["paths"],
            result["y_true"],
            result["y_pred"],
            names,
            output_dir / f"cnn_{split}_misclassified.png",
        )

    print(f"Best epoch: {best_epoch}, best val F1: {best_score:.4f}")
    print(f"Saved CNN artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
