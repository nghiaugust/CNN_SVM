from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from src.config import label_names, load_config
from src.data import make_eval_loader
from src.engine import evaluate_cnn
from src.metrics import save_classification_outputs
from src.models import load_cnn_checkpoint
from src.utils import configure_device, describe_device, ensure_dir, get_device
from src.visualize import plot_binary_curves, plot_confusion, plot_misclassified_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN or SVM on train/val/test splits.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-type", choices=["cnn", "svm"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", default=None, help="CNN .pt checkpoint when model-type=cnn.")
    parser.add_argument("--svm-model", default=None, help="svm_model.joblib file when model-type=svm.")
    parser.add_argument("--feature-cache", default=None, help="features.npz file created by train_svm.py.")
    parser.add_argument("--output-dir", default="runs/evaluation")
    parser.add_argument("--device", default=None, help="Device to use for CNN evaluation: auto, cpu, cuda, cuda:0, ...")
    return parser.parse_args()


def evaluate_cnn_cli(cfg: dict, args: argparse.Namespace, names: list[str]) -> None:
    checkpoint = args.checkpoint or cfg["svm"].get("cnn_checkpoint") or "runs/cnn_resnet18/best_cnn.pt"
    output_dir = ensure_dir(args.output_dir)
    device = get_device(str(cfg.get("device", "auto")))
    configure_device(device)
    print(f"Device: {describe_device(device)}")
    model, _ = load_cnn_checkpoint(checkpoint, num_classes=len(names), device=device)
    loader = make_eval_loader(cfg, args.split)
    result = evaluate_cnn(
        model,
        loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        amp=bool(cfg["training"].get("amp", True)),
        desc=f"cnn {args.split}",
    )
    prefix = f"cnn_{args.split}"
    metrics = save_classification_outputs(
        result["y_true"],
        result["y_pred"],
        names,
        output_dir,
        prefix,
        paths=result["paths"],
        probabilities=result["probabilities"],
    )
    plot_confusion(result["y_true"], result["y_pred"], names, output_dir / f"{prefix}_confusion.png", prefix)
    plot_binary_curves(result["y_true"], result["probabilities"], output_dir, prefix, positive_label=1)
    plot_misclassified_grid(result["paths"], result["y_true"], result["y_pred"], names, output_dir / f"{prefix}_misclassified.png")
    print(f"{prefix}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['f1_macro']:.4f}")


def evaluate_svm_cli(cfg: dict, args: argparse.Namespace, names: list[str]) -> None:
    svm_model = Path(args.svm_model or Path(cfg["svm"]["output_dir"]) / "svm_model.joblib")
    feature_cache = Path(args.feature_cache or cfg["svm"]["feature_cache"])
    if not feature_cache.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {feature_cache}. Run train_svm.py first, "
            "or pass the correct --feature-cache."
        )
    output_dir = ensure_dir(args.output_dir)
    model = joblib.load(svm_model)
    data = np.load(feature_cache, allow_pickle=True)
    X = data[f"X_{args.split}"]
    y = data[f"y_{args.split}"]
    paths = data[f"paths_{args.split}"].astype(str).tolist()
    pred = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    prefix = f"svm_{args.split}"
    metrics = save_classification_outputs(y, pred, names, output_dir, prefix, paths=paths, probabilities=probabilities)
    plot_confusion(y, pred, names, output_dir / f"{prefix}_confusion.png", prefix)
    plot_binary_curves(y, probabilities, output_dir, prefix, positive_label=1)
    plot_misclassified_grid(paths, y, pred, names, output_dir / f"{prefix}_misclassified.png")
    print(f"{prefix}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['f1_macro']:.4f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    names = label_names(cfg)
    if args.model_type == "cnn":
        evaluate_cnn_cli(cfg, args, names)
    else:
        evaluate_svm_cli(cfg, args, names)


if __name__ == "__main__":
    main()
