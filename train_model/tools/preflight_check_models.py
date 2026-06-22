from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config import label_names, load_config
from src.data import NameAnnotationDataset, build_transform
from src.models import CNNFeatureExtractor, build_model, get_feature_dim, normalize_model_name
from src.utils import configure_device, describe_device, get_device


MODEL_CONFIGS = {
    "resnet18": Path("configs/config.yaml"),
    "resnet50": Path("configs/config_resnet50.yaml"),
    "convnext_tiny": Path("configs/config_convnext_tiny.yaml"),
    "deit_small": Path("configs/config_deit_small.yaml"),
    "yolov8": Path("configs/config_yolov8.yaml"),
}

CNN_SVM_MODELS = {"resnet18", "resnet50", "convnext_tiny", "deit_small"}


@dataclass
class CheckResult:
    model: str
    status: str
    seconds: float
    batch_shape: str = ""
    output_shape: str = ""
    feature_shape: str = ""
    loss: float | None = None
    svm_tiny_fit: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick preflight checks before training all models.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or all. Supported: resnet18,resnet50,convnext_tiny,deit_small,yolov8.",
    )
    parser.add_argument("--dataset-root", default="auto", help="Default auto uses config root, then data/dataset fallback.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--pretrained", action="store_true", help="Also test pretrained weight loading.")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward/optimizer step for CNN backbones.")
    parser.add_argument("--skip-yolo-forward", action="store_true", help="Only test Ultralytics import and model load.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default="runs/preflight/preflight_results.csv")
    return parser.parse_args()


def parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(MODEL_CONFIGS)
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in models if name not in MODEL_CONFIGS]
    if unknown:
        supported = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}. Supported: {supported}")
    return models


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_dataset_root(config_root: str | Path, override: str | Path) -> Path:
    if str(override).strip().lower() != "auto":
        root = project_path(override)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        return root

    root = project_path(config_root)
    if root.exists():
        return root

    fallback = PROJECT_ROOT / "data" / Path(config_root).name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Dataset root not found. Tried config root '{root}' and fallback '{fallback}'. "
        "Pass --dataset-root explicitly if your dataset is elsewhere."
    )


def load_checked_config(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(project_path(MODEL_CONFIGS[model_name]))
    dataset_root = resolve_dataset_root(cfg["dataset"]["root"], args.dataset_root)
    cfg["dataset"]["root"] = relative_to_project(dataset_root)
    for key in ("train_annotation", "val_annotation", "test_annotation"):
        annotation_path = dataset_root / cfg["dataset"][key]
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing {key}: {annotation_path}")
    return cfg


def select_diverse_indices(dataset: NameAnnotationDataset, batch_size: int, num_classes: int) -> list[int]:
    selected: list[int] = []
    seen_labels: set[int] = set()
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in seen_labels:
            selected.append(idx)
            seen_labels.add(label)
            if len(selected) >= min(batch_size, num_classes):
                break

    for idx in range(len(dataset)):
        if len(selected) >= batch_size:
            break
        if idx not in selected:
            selected.append(idx)
    return selected


def make_batch(cfg: dict[str, Any], split: str, batch_size: int):
    ds_cfg = cfg["dataset"]
    transform = build_transform(
        input_size=ds_cfg["input_size"],
        train=split == "train",
        augment=bool(cfg.get("training", {}).get("augment", True)),
        pad_color=int(ds_cfg.get("pad_color", 255)),
    )
    annotation_file = ds_cfg[f"{split}_annotation"]
    dataset = NameAnnotationDataset(ds_cfg["root"], annotation_file, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No samples found for split: {split}")
    indices = select_diverse_indices(dataset, min(batch_size, len(dataset)), len(ds_cfg["label_names"]))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(indices), shuffle=False, num_workers=0)
    images, labels, paths = next(iter(loader))
    return images, labels, paths


def shape_text(value) -> str:
    if hasattr(value, "shape"):
        return "x".join(str(v) for v in value.shape)
    return str(type(value).__name__)


def check_cnn_svm_model(model_name: str, cfg: dict[str, Any], args: argparse.Namespace, device: torch.device) -> CheckResult:
    start = time.perf_counter()
    names = label_names(cfg)
    images, labels, _ = make_batch(cfg, args.split, args.batch_size)
    images = images.to(device)
    labels = labels.to(device)

    normalized_name = normalize_model_name(str(cfg["model"]["name"]))
    model = build_model(
        normalized_name,
        num_classes=len(names),
        pretrained=bool(args.pretrained and cfg["model"].get("pretrained", True)),
        input_size=cfg["dataset"].get("input_size"),
    )
    model.to(device)
    model.train()

    logits = model(images)
    if logits.ndim != 2 or logits.shape[0] != images.shape[0] or logits.shape[1] != len(names):
        raise RuntimeError(f"Unexpected output shape for {model_name}: {tuple(logits.shape)}")

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    if not args.no_backward:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        extractor = CNNFeatureExtractor(model, normalized_name).to(device)
        features = extractor(images)
    expected_dim = get_feature_dim(normalized_name)
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected feature shape for {model_name}: {tuple(features.shape)}")

    svm_status = "skipped"
    unique_labels = labels.detach().cpu().unique().tolist()
    if len(unique_labels) >= 2:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="linear", class_weight="balanced")),
            ]
        )
        pipeline.fit(features.detach().cpu().numpy().astype("float32"), labels.detach().cpu().numpy())
        svm_status = "ok"

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return CheckResult(
        model=model_name,
        status="ok",
        seconds=time.perf_counter() - start,
        batch_shape=shape_text(images),
        output_shape=shape_text(logits),
        feature_shape=shape_text(features),
        loss=float(loss.detach().cpu().item()),
        svm_tiny_fit=svm_status,
    )


def normalize_yolo_outputs(outputs) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    if not torch.is_tensor(outputs):
        raise TypeError(f"Unexpected YOLO output type: {type(outputs).__name__}")
    row_sums = outputs.sum(dim=1)
    if outputs.min().item() < 0 or not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        return torch.softmax(outputs, dim=1)
    return outputs


def check_yolov8(cfg: dict[str, Any], args: argparse.Namespace, device: torch.device) -> CheckResult:
    start = time.perf_counter()
    images, _, _ = make_batch(cfg, args.split, args.batch_size)
    images = images.to(device)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Missing dependency 'ultralytics'. Install it with: pip install -r requirements.txt") from exc

    yolo = YOLO(str(cfg["model"]["name"]))
    model = yolo.model.to(device)
    model.eval()

    output_shape = "not_run"
    if not args.skip_yolo_forward:
        with torch.no_grad():
            outputs = model(images)
            probabilities = normalize_yolo_outputs(outputs)
            output_shape = shape_text(probabilities)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return CheckResult(
        model="yolov8",
        status="ok",
        seconds=time.perf_counter() - start,
        batch_shape=shape_text(images),
        output_shape=output_shape,
    )


def result_to_dict(result: CheckResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "status": result.status,
        "seconds": result.seconds,
        "batch_shape": result.batch_shape,
        "output_shape": result.output_shape,
        "feature_shape": result.feature_shape,
        "loss": result.loss,
        "svm_tiny_fit": result.svm_tiny_fit,
        "error": result.error,
    }


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    device = get_device(args.device)
    configure_device(device)
    print(f"Device: {describe_device(device)}")
    print(f"Models: {', '.join(models)}")
    if args.pretrained:
        print("Pretrained loading is enabled; first run may download weights.")

    rows: list[dict[str, Any]] = []
    for model_name in models:
        print(f"\nCheck {model_name}...")
        try:
            cfg = load_checked_config(model_name, args)
            if model_name in CNN_SVM_MODELS:
                result = check_cnn_svm_model(model_name, cfg, args, device)
            else:
                result = check_yolov8(cfg, args, device)
            rows.append(result_to_dict(result))
            print(
                f"OK {model_name}: output={result.output_shape}, "
                f"features={result.feature_shape or '-'}, seconds={result.seconds:.2f}"
            )
        except Exception as exc:
            rows.append(
                result_to_dict(
                    CheckResult(
                        model=model_name,
                        status="error",
                        seconds=0.0,
                        error=str(exc),
                    )
                )
            )
            print(f"ERROR {model_name}: {exc}")
            if args.fail_fast:
                break

    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f"\nSaved preflight results to: {output_path}")
    print(results.to_string(index=False))

    if (results["status"] != "ok").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
