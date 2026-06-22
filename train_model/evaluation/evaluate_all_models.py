from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config import label_names, load_config
from src.data import build_transform
from src.models import CNNFeatureExtractor, build_model, normalize_model_name
from src.utils import configure_device, cpu_count_for_loader, describe_device, get_device


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_path: Path
    kind: str


@dataclass(frozen=True)
class EvalDatasetSpec:
    name: str
    samples: list[tuple[Path, int]]


MODEL_SPECS = {
    "resnet18": ModelSpec("resnet18", Path("configs/config.yaml"), "cnn_svm"),
    "resnet50": ModelSpec("resnet50", Path("configs/config_resnet50.yaml"), "cnn_svm"),
    "convnext_tiny": ModelSpec("convnext_tiny", Path("configs/config_convnext_tiny.yaml"), "cnn_svm"),
    "deit_small": ModelSpec("deit_small", Path("configs/config_deit_small.yaml"), "cnn_svm"),
    "yolov8": ModelSpec("yolov8", Path("configs/config_yolov8.yaml"), "yolo"),
}


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained CNN/SVM/YOLO models on test annotation and data/evaluation."
    )
    parser.add_argument("--models", default="all", help="Comma-separated models or all.")
    parser.add_argument("--datasets", default="test,evaluation", help="Comma-separated datasets: test,evaluation.")
    parser.add_argument("--cnn-modes", default="cnn,svm", help="For CNN backbones: cnn,svm or both comma-separated.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--positive-label", type=int, default=0, help="Positive class for precision/recall/f1. Default 0 = Gach_Ten.")
    parser.add_argument("--output-dir", default="runs/evaluation_all")
    parser.add_argument("--metrics-file", default="metrics.csv")
    parser.add_argument("--predictions-file", default="predictions.csv")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only list datasets and expected weight paths.")
    return parser.parse_args()


def parse_csv_list(raw: str, choices: Iterable[str] | None = None) -> list[str]:
    if raw.strip().lower() == "all" and choices is not None:
        return list(choices)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if choices is not None:
        unknown = [value for value in values if value not in choices]
        if unknown:
            raise ValueError(f"Unsupported value(s): {', '.join(unknown)}. Supported: {', '.join(choices)}")
    return values


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_dataset_root(config_root: str | Path) -> Path:
    root = resolve_project_path(config_root)
    if root.exists():
        return root

    fallback = PROJECT_ROOT / "data" / Path(config_root).name
    if fallback.exists():
        return fallback

    return root


def natural_key(path: Path) -> list[Any]:
    parts: list[Any] = []
    current = ""
    for char in path.as_posix():
        if char.isdigit():
            current += char
        else:
            if current:
                parts.append(int(current))
                current = ""
            parts.append(char)
    if current:
        parts.append(int(current))
    return parts


def read_annotation_samples(root: Path, annotation_file: str | Path) -> list[tuple[Path, int]]:
    annotation_path = root / annotation_file
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    samples: list[tuple[Path, int]] = []
    for line_no, raw in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid annotation at {annotation_path}:{line_no}: {raw}")
        rel_path, label_text = parts
        image_path = root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {annotation_path}:{line_no}: {image_path}")
        samples.append((image_path, int(label_text)))
    return samples


def normalize_label_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def label_from_folder(folder_name: str, names: list[str]) -> int | None:
    text = normalize_label_name(folder_name)
    if text.isdigit():
        value = int(text)
        return value if 0 <= value < len(names) else None

    name_to_label = {normalize_label_name(name): idx for idx, name in enumerate(names)}
    if text in name_to_label:
        return name_to_label[text]

    aliases = {
        "gachten": "gach_ten",
        "co_gach": "gach_ten",
        "cogach": "gach_ten",
        "crossed": "gach_ten",
        "ten": "ten",
        "khong_gach": "ten",
        "khonggach": "ten",
        "no_cross": "ten",
    }
    mapped = aliases.get(text)
    if mapped is not None:
        return name_to_label.get(mapped)
    return None


def read_class_folder_samples(root: Path, names: list[str]) -> list[tuple[Path, int]]:
    if not root.exists():
        raise FileNotFoundError(f"Evaluation folder not found: {root}")

    samples: list[tuple[Path, int]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=natural_key):
        label = label_from_folder(class_dir.name, names)
        if label is None:
            print(f"[WARN] Skip unknown class folder: {class_dir}")
            continue
        image_paths = [
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image_path in sorted(image_paths, key=natural_key):
            samples.append((image_path, label))

    if not samples:
        raise ValueError(f"No evaluation images found in: {root}")
    return samples


def build_eval_datasets(base_cfg: dict, requested: list[str]) -> list[EvalDatasetSpec]:
    names = label_names(base_cfg)
    ds_cfg = base_cfg["dataset"]
    dataset_root = resolve_dataset_root(ds_cfg["root"])

    datasets: list[EvalDatasetSpec] = []
    if "test" in requested:
        datasets.append(
            EvalDatasetSpec(
                name="test",
                samples=read_annotation_samples(dataset_root, ds_cfg["test_annotation"]),
            )
        )
    if "evaluation" in requested:
        datasets.append(
            EvalDatasetSpec(
                name="evaluation",
                samples=read_class_folder_samples(PROJECT_ROOT / "data" / "evaluation", names),
            )
        )
    return datasets


def build_loader(samples: list[tuple[Path, int]], cfg: dict, batch_size: int, num_workers: int) -> DataLoader:
    ds_cfg = cfg["dataset"]
    transform = build_transform(
        input_size=ds_cfg["input_size"],
        train=False,
        augment=False,
        pad_color=int(ds_cfg.get("pad_color", 255)),
    )
    dataset = ImagePathDataset(samples, transform=transform)
    workers = cpu_count_for_loader(num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_cnn_model(checkpoint_path: Path, cfg: dict, names: list[str], device: torch.device) -> tuple[torch.nn.Module, str]:
    checkpoint = torch_load(checkpoint_path, device)
    checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_section = checkpoint_cfg.get("model", {}) if isinstance(checkpoint_cfg, dict) else {}
    dataset_section = checkpoint_cfg.get("dataset", {}) if isinstance(checkpoint_cfg, dict) else {}
    if not isinstance(model_section, dict):
        model_section = {}
    if not isinstance(dataset_section, dict):
        dataset_section = {}

    model_name = None
    if isinstance(checkpoint, dict):
        model_name = checkpoint.get("model_name") or model_section.get("name")
    model_name = normalize_model_name(str(model_name or cfg["model"]["name"]))
    input_size = dataset_section.get("input_size") or cfg["dataset"].get("input_size")
    model = build_model(model_name, num_classes=len(names), pretrained=False, input_size=input_size)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_inference(device: torch.device, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return result, time.perf_counter() - start


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def predict_cnn(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, list[str], float, float]:
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    inference_time = 0.0
    wall_start = time.perf_counter()

    for images, labels, batch_paths in tqdm(loader, desc="cnn", leave=False):
        images = images.to(device, non_blocking=True)

        def forward():
            with autocast_context(device, amp):
                logits = model(images)
            return logits.argmax(dim=1).detach().cpu().numpy()

        preds, elapsed = timed_inference(device, forward)
        inference_time += elapsed
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.astype(int).tolist())
        paths.extend(list(batch_paths))

    return np.asarray(y_true), np.asarray(y_pred), paths, inference_time, time.perf_counter() - wall_start


@torch.no_grad()
def predict_svm(
    model: torch.nn.Module,
    svm_model,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], float, float]:
    extractor = CNNFeatureExtractor(model).to(device)
    extractor.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    inference_time = 0.0
    wall_start = time.perf_counter()

    for images, labels, batch_paths in tqdm(loader, desc="svm", leave=False):
        images = images.to(device, non_blocking=True)

        def forward():
            with autocast_context(device, amp):
                features = extractor(images)
            features_np = features.detach().cpu().numpy().astype("float32")
            return svm_model.predict(features_np)

        preds, elapsed = timed_inference(device, forward)
        inference_time += elapsed
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(np.asarray(preds).astype(int).tolist())
        paths.extend(list(batch_paths))

    return np.asarray(y_true), np.asarray(y_pred), paths, inference_time, time.perf_counter() - wall_start


def normalize_yolo_outputs(outputs) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    probabilities = outputs
    row_sums = probabilities.sum(dim=1)
    if probabilities.min().item() < 0 or not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        probabilities = torch.softmax(probabilities, dim=1)
    return probabilities


@torch.no_grad()
def predict_yolo(yolo_model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], float, float]:
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    inference_time = 0.0
    wall_start = time.perf_counter()

    for images, labels, batch_paths in tqdm(loader, desc="yolov8", leave=False):
        images = images.to(device, non_blocking=True)

        def forward():
            outputs = yolo_model(images)
            probabilities = normalize_yolo_outputs(outputs)
            return probabilities.argmax(dim=1).detach().cpu().numpy()

        preds, elapsed = timed_inference(device, forward)
        inference_time += elapsed
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(np.asarray(preds).astype(int).tolist())
        paths.extend(list(batch_paths))

    return np.asarray(y_true), np.asarray(y_pred), paths, inference_time, time.perf_counter() - wall_start


def metric_row(
    model_name: str,
    mode: str,
    dataset_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names_: list[str],
    positive_label: int,
    inference_time: float,
    wall_time: float,
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    labels = list(range(len(label_names_)))
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    num_images = int(len(y_true))
    pos = int(positive_label)
    row = {
        "model": model_name,
        "mode": mode,
        "dataset": dataset_name,
        "status": status,
        "num_images": num_images,
        "positive_label": pos,
        "positive_name": label_names_[pos] if 0 <= pos < len(label_names_) else str(pos),
        "precision": float(per_precision[pos]) if 0 <= pos < len(per_precision) else np.nan,
        "recall": float(per_recall[pos]) if 0 <= pos < len(per_recall) else np.nan,
        "f1": float(per_f1[pos]) if 0 <= pos < len(per_f1) else np.nan,
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "support_0": int(support[0]) if len(support) > 0 else 0,
        "support_1": int(support[1]) if len(support) > 1 else 0,
        "confusion_matrix": cm.tolist(),
        "inference_time_s": float(inference_time),
        "wall_time_s": float(wall_time),
        "time_per_image_ms": float((inference_time / max(num_images, 1)) * 1000.0),
        "images_per_second": float(num_images / inference_time) if inference_time > 0 else 0.0,
        "error": error,
    }
    for idx, name in enumerate(label_names_):
        row[f"precision_{name}"] = float(per_precision[idx])
        row[f"recall_{name}"] = float(per_recall[idx])
        row[f"f1_{name}"] = float(per_f1[idx])
        row[f"support_{name}"] = int(support[idx])
    return row


def missing_row(model_name: str, mode: str, dataset_name: str, num_images: int, error: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "mode": mode,
        "dataset": dataset_name,
        "status": "missing_weights",
        "num_images": int(num_images),
        "error": error,
    }


def error_row(model_name: str, mode: str, dataset_name: str, num_images: int, error: Exception) -> dict[str, Any]:
    return {
        "model": model_name,
        "mode": mode,
        "dataset": dataset_name,
        "status": "error",
        "num_images": int(num_images),
        "error": str(error),
    }


def prediction_frame(model_name: str, mode: str, dataset_name: str, paths: list[str], y_true, y_pred, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "mode": mode,
            "dataset": dataset_name,
            "path": paths,
            "y_true": y_true,
            "y_pred": y_pred,
            "true_name": [names[int(v)] for v in y_true],
            "pred_name": [names[int(v)] for v in y_pred],
            "correct": [int(t) == int(p) for t, p in zip(y_true, y_pred)],
        }
    )


def cnn_checkpoint_path(cfg: dict) -> Path:
    return resolve_project_path(Path(cfg["training"]["output_dir"]) / "best_cnn.pt")


def svm_model_path(cfg: dict) -> Path:
    return resolve_project_path(Path(cfg["svm"]["output_dir"]) / "svm_model.joblib")


def yolo_weights_path(cfg: dict) -> Path:
    run_dir = resolve_project_path(Path(cfg["training"]["output_dir"]) / str(cfg["training"]["run_name"]))
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        return best
    return run_dir / "weights" / "last.pt"


def yolo_weight_candidates(cfg: dict) -> tuple[Path, Path]:
    run_dir = resolve_project_path(Path(cfg["training"]["output_dir"]) / str(cfg["training"]["run_name"]))
    return run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"


def evaluate_cnn_svm_model(
    spec: ModelSpec,
    cfg: dict,
    datasets: list[EvalDatasetSpec],
    modes: list[str],
    device: torch.device,
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
) -> None:
    names = label_names(cfg)
    checkpoint_path = cnn_checkpoint_path(cfg)
    svm_path = svm_model_path(cfg)

    for dataset_spec in datasets:
        batch_size = args.batch_size or int(cfg["training"]["batch_size"])
        loader = build_loader(dataset_spec.samples, cfg, batch_size=batch_size, num_workers=args.num_workers)

        for mode in modes:
            if mode == "cnn" and not checkpoint_path.exists():
                metric_rows.append(missing_row(spec.name, mode, dataset_spec.name, len(dataset_spec.samples), f"Missing CNN checkpoint: {checkpoint_path}"))
                continue
            if mode == "svm" and (not checkpoint_path.exists() or not svm_path.exists()):
                missing = []
                if not checkpoint_path.exists():
                    missing.append(f"CNN checkpoint: {checkpoint_path}")
                if not svm_path.exists():
                    missing.append(f"SVM model: {svm_path}")
                metric_rows.append(missing_row(spec.name, mode, dataset_spec.name, len(dataset_spec.samples), "Missing " + "; ".join(missing)))
                continue

            try:
                model, _ = load_cnn_model(checkpoint_path, cfg, names, device)
                amp = bool(cfg["training"].get("amp", True))
                print(f"Evaluate {spec.name}:{mode} on {dataset_spec.name} ({len(dataset_spec.samples)} images)")
                if mode == "cnn":
                    y_true, y_pred, paths, infer_time, wall_time = predict_cnn(model, loader, device, amp)
                else:
                    svm_model = joblib.load(svm_path)
                    y_true, y_pred, paths, infer_time, wall_time = predict_svm(model, svm_model, loader, device, amp)

                metric_rows.append(
                    metric_row(
                        spec.name,
                        mode,
                        dataset_spec.name,
                        y_true,
                        y_pred,
                        names,
                        args.positive_label,
                        infer_time,
                        wall_time,
                    )
                )
                if not args.no_save_predictions:
                    prediction_frames.append(prediction_frame(spec.name, mode, dataset_spec.name, paths, y_true, y_pred, names))
            except Exception as exc:
                metric_rows.append(error_row(spec.name, mode, dataset_spec.name, len(dataset_spec.samples), exc))


def evaluate_yolo_model(
    spec: ModelSpec,
    cfg: dict,
    datasets: list[EvalDatasetSpec],
    device: torch.device,
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
) -> None:
    names = label_names(cfg)
    weights_path = yolo_weights_path(cfg)
    best_path, last_path = yolo_weight_candidates(cfg)

    for dataset_spec in datasets:
        if not weights_path.exists():
            metric_rows.append(
                missing_row(
                    spec.name,
                    "classify",
                    dataset_spec.name,
                    len(dataset_spec.samples),
                    f"Missing YOLO weights: best={best_path}; last={last_path}",
                )
            )
            continue

        try:
            from ultralytics import YOLO

            batch_size = args.batch_size or int(cfg["training"]["batch_size"])
            loader = build_loader(dataset_spec.samples, cfg, batch_size=batch_size, num_workers=args.num_workers)
            yolo = YOLO(str(weights_path))
            model = yolo.model.to(device)
            model.eval()

            print(f"Evaluate {spec.name}:classify on {dataset_spec.name} ({len(dataset_spec.samples)} images)")
            y_true, y_pred, paths, infer_time, wall_time = predict_yolo(model, loader, device)
            metric_rows.append(
                metric_row(
                    spec.name,
                    "classify",
                    dataset_spec.name,
                    y_true,
                    y_pred,
                    names,
                    args.positive_label,
                    infer_time,
                    wall_time,
                )
            )
            if not args.no_save_predictions:
                prediction_frames.append(prediction_frame(spec.name, "classify", dataset_spec.name, paths, y_true, y_pred, names))
        except Exception as exc:
            metric_rows.append(error_row(spec.name, "classify", dataset_spec.name, len(dataset_spec.samples), exc))


def dry_run_report(models: list[str], datasets: list[EvalDatasetSpec], modes: list[str]) -> None:
    print("Datasets:")
    for dataset in datasets:
        counts = pd.Series([label for _, label in dataset.samples]).value_counts().sort_index().to_dict()
        print(f"  {dataset.name}: {len(dataset.samples)} images, counts={counts}")

    print("Model artifacts:")
    for model_name in models:
        spec = MODEL_SPECS[model_name]
        cfg = load_config(resolve_project_path(spec.config_path))
        if spec.kind == "yolo":
            best_path, last_path = yolo_weight_candidates(cfg)
            print(
                f"  {model_name}: best={best_path} exists={best_path.exists()} | "
                f"last={last_path} exists={last_path.exists()}"
            )
            continue
        checkpoint = cnn_checkpoint_path(cfg)
        svm_path = svm_model_path(cfg)
        for mode in modes:
            if mode == "cnn":
                print(f"  {model_name}: cnn={checkpoint} exists={checkpoint.exists()}")
            elif mode == "svm":
                print(f"  {model_name}: cnn={checkpoint} exists={checkpoint.exists()} | svm={svm_path} exists={svm_path.exists()}")


def main() -> None:
    args = parse_args()
    models = parse_csv_list(args.models, MODEL_SPECS.keys())
    datasets_requested = parse_csv_list(args.datasets, {"test", "evaluation"})
    modes = parse_csv_list(args.cnn_modes, {"cnn", "svm"})

    base_cfg = load_config(resolve_project_path("configs/config.yaml"))
    datasets = build_eval_datasets(base_cfg, datasets_requested)

    if args.dry_run:
        dry_run_report(models, datasets, modes)
        return

    device = get_device(args.device)
    configure_device(device)
    print(f"Device: {describe_device(device)}")

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name in models:
        spec = MODEL_SPECS[model_name]
        cfg = load_config(resolve_project_path(spec.config_path))
        if spec.kind == "yolo":
            evaluate_yolo_model(spec, cfg, datasets, device, args, metric_rows, prediction_frames)
        else:
            evaluate_cnn_svm_model(spec, cfg, datasets, modes, device, args, metric_rows, prediction_frames)

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / args.metrics_file
    metrics.to_csv(metrics_path, index=False)

    if prediction_frames and not args.no_save_predictions:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions_path = output_dir / args.predictions_file
        predictions.to_csv(predictions_path, index=False)
        print(f"Saved predictions to: {predictions_path}")

    display_columns = [
        "model",
        "mode",
        "dataset",
        "status",
        "num_images",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "time_per_image_ms",
        "inference_time_s",
        "error",
    ]
    for column in display_columns:
        if column not in metrics.columns:
            metrics[column] = np.nan
    print(metrics[display_columns].to_string(index=False))
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
