from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import torch

from evaluate_all_models import (
    MODEL_SPECS,
    ModelSpec,
    autocast_context,
    build_eval_datasets,
    build_loader,
    cnn_checkpoint_path,
    load_cnn_model,
    normalize_yolo_outputs,
    parse_csv_list,
    resolve_project_path,
    svm_model_path,
    sync_if_needed,
    yolo_weight_candidates,
    yolo_weights_path,
)
from src.config import label_names, load_config
from src.models import CNNFeatureExtractor
from src.utils import configure_device, describe_device, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end inference time for trained CNN/SVM/YOLO models. "
            "The benchmark excludes model loading, runs warm-up passes, then repeats "
            "full dataloader inference multiple times."
        )
    )
    parser.add_argument("--models", default="all", help="Comma-separated models or all.")
    parser.add_argument("--datasets", default="test,evaluation", help="Comma-separated datasets: test,evaluation.")
    parser.add_argument("--cnn-modes", default="cnn,svm", help="For CNN backbones: cnn,svm or both comma-separated.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warm-up runs not included in the final statistics.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs used for statistics.")
    parser.add_argument("--output-dir", default="runs/evaluation_all")
    parser.add_argument("--output-file", default="benchmark_time.csv")
    parser.add_argument("--txt-file", default="benchmark_time.txt")
    parser.add_argument("--dry-run", action="store_true", help="Only list benchmark targets and artifact paths.")
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.6f}"


def timed_full_pass(fn: Callable[[], int], device: torch.device) -> tuple[float, int]:
    sync_if_needed(device)
    start = time.perf_counter()
    num_images = fn()
    sync_if_needed(device)
    return time.perf_counter() - start, num_images


@torch.no_grad()
def run_cnn_once(model: torch.nn.Module, loader, device: torch.device, amp: bool) -> int:
    model.eval()
    total = 0
    for images, _labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(images)
        preds = logits.argmax(dim=1)
        total += int(preds.numel())
        preds.detach().cpu().numpy()
    return total


@torch.no_grad()
def run_svm_once(
    extractor: torch.nn.Module,
    svm_model,
    loader,
    device: torch.device,
    amp: bool,
) -> int:
    extractor.eval()
    total = 0
    for images, _labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device, amp):
            features = extractor(images)
        features_np = features.detach().cpu().numpy().astype("float32")
        preds = svm_model.predict(features_np)
        total += int(len(preds))
    return total


@torch.no_grad()
def run_yolo_once(yolo_model, loader, device: torch.device) -> int:
    yolo_model.eval()
    total = 0
    for images, _labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        outputs = yolo_model(images)
        probabilities = normalize_yolo_outputs(outputs)
        preds = probabilities.argmax(dim=1)
        total += int(preds.numel())
        preds.detach().cpu().numpy()
    return total


def summarize_times(times: list[float], num_images: int) -> dict[str, Any]:
    mean_s = statistics.mean(times)
    median_s = statistics.median(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0
    min_s = min(times)
    max_s = max(times)
    return {
        "mean_wall_time_s": mean_s,
        "median_wall_time_s": median_s,
        "std_wall_time_s": std_s,
        "min_wall_time_s": min_s,
        "max_wall_time_s": max_s,
        "mean_time_per_image_ms": mean_s * 1000.0 / max(num_images, 1),
        "median_time_per_image_ms": median_s * 1000.0 / max(num_images, 1),
        "min_time_per_image_ms": min_s * 1000.0 / max(num_images, 1),
        "max_time_per_image_ms": max_s * 1000.0 / max(num_images, 1),
        "images_per_second_mean": num_images / mean_s if mean_s > 0 else 0.0,
        "per_run_wall_time_s": ";".join(fmt(value) for value in times),
    }


def benchmark_callable(
    runner: Callable[[], int],
    device: torch.device,
    warmup_runs: int,
    measured_runs: int,
    expected_images: int,
) -> tuple[list[float], int]:
    for _ in range(max(warmup_runs, 0)):
        _elapsed, seen = timed_full_pass(runner, device)
        if seen != expected_images:
            raise RuntimeError(f"Warm-up processed {seen} images, expected {expected_images}.")

    times: list[float] = []
    final_seen = 0
    for _ in range(max(measured_runs, 1)):
        elapsed, seen = timed_full_pass(runner, device)
        if seen != expected_images:
            raise RuntimeError(f"Measured run processed {seen} images, expected {expected_images}.")
        times.append(elapsed)
        final_seen = seen
    return times, final_seen


def missing_row(
    model_name: str,
    mode: str,
    dataset_name: str,
    num_images: int,
    batch_size: int,
    num_workers: int,
    warmup_runs: int,
    measured_runs: int,
    error: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "mode": mode,
        "dataset": dataset_name,
        "status": "missing_artifact",
        "num_images": num_images,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_wall_time_s": "",
        "median_wall_time_s": "",
        "std_wall_time_s": "",
        "min_wall_time_s": "",
        "max_wall_time_s": "",
        "mean_time_per_image_ms": "",
        "median_time_per_image_ms": "",
        "min_time_per_image_ms": "",
        "max_time_per_image_ms": "",
        "images_per_second_mean": "",
        "per_run_wall_time_s": "",
        "error": error,
    }


def result_row(
    model_name: str,
    mode: str,
    dataset_name: str,
    num_images: int,
    batch_size: int,
    num_workers: int,
    warmup_runs: int,
    measured_runs: int,
    times: list[float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "mode": mode,
        "dataset": dataset_name,
        "status": "ok",
        "num_images": num_images,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "error": "",
    }
    row.update(summarize_times(times, num_images))
    return row


def benchmark_cnn_svm(
    spec: ModelSpec,
    cfg: dict,
    datasets,
    modes: list[str],
    device: torch.device,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    names = label_names(cfg)
    checkpoint_path = cnn_checkpoint_path(cfg)
    svm_path = svm_model_path(cfg)
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    amp = bool(cfg["training"].get("amp", True))

    for dataset_spec in datasets:
        loader = build_loader(dataset_spec.samples, cfg, batch_size=batch_size, num_workers=args.num_workers)
        expected_images = len(dataset_spec.samples)

        for mode in modes:
            print(f"Benchmark {spec.name}:{mode} on {dataset_spec.name} ({expected_images} images)")

            if not checkpoint_path.exists():
                rows.append(
                    missing_row(
                        spec.name,
                        mode,
                        dataset_spec.name,
                        expected_images,
                        batch_size,
                        args.num_workers,
                        args.warmup_runs,
                        args.runs,
                        f"Missing CNN checkpoint: {checkpoint_path}",
                    )
                )
                continue
            if mode == "svm" and not svm_path.exists():
                rows.append(
                    missing_row(
                        spec.name,
                        mode,
                        dataset_spec.name,
                        expected_images,
                        batch_size,
                        args.num_workers,
                        args.warmup_runs,
                        args.runs,
                        f"Missing SVM model: {svm_path}",
                    )
                )
                continue

            model, _model_name = load_cnn_model(checkpoint_path, cfg, names, device)
            if mode == "cnn":
                runner = lambda: run_cnn_once(model, loader, device, amp)
            else:
                svm_model = joblib.load(svm_path)
                extractor = CNNFeatureExtractor(model).to(device)
                extractor.eval()
                runner = lambda: run_svm_once(extractor, svm_model, loader, device, amp)

            times, seen = benchmark_callable(
                runner,
                device,
                args.warmup_runs,
                args.runs,
                expected_images,
            )
            rows.append(
                result_row(
                    spec.name,
                    mode,
                    dataset_spec.name,
                    seen,
                    batch_size,
                    args.num_workers,
                    args.warmup_runs,
                    args.runs,
                    times,
                )
            )


def benchmark_yolo(
    spec: ModelSpec,
    cfg: dict,
    datasets,
    device: torch.device,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    weights_path = yolo_weights_path(cfg)
    best_path, last_path = yolo_weight_candidates(cfg)
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])

    for dataset_spec in datasets:
        expected_images = len(dataset_spec.samples)
        print(f"Benchmark {spec.name}:classify on {dataset_spec.name} ({expected_images} images)")

        if not weights_path.exists():
            rows.append(
                missing_row(
                    spec.name,
                    "classify",
                    dataset_spec.name,
                    expected_images,
                    batch_size,
                    args.num_workers,
                    args.warmup_runs,
                    args.runs,
                    f"Missing YOLO weights: best={best_path}; last={last_path}",
                )
            )
            continue

        from ultralytics import YOLO

        loader = build_loader(dataset_spec.samples, cfg, batch_size=batch_size, num_workers=args.num_workers)
        yolo = YOLO(str(weights_path))
        model = yolo.model.to(device)
        model.eval()
        runner = lambda: run_yolo_once(model, loader, device)
        times, seen = benchmark_callable(runner, device, args.warmup_runs, args.runs, expected_images)
        rows.append(
            result_row(
                spec.name,
                "classify",
                dataset_spec.name,
                seen,
                batch_size,
                args.num_workers,
                args.warmup_runs,
                args.runs,
                times,
            )
        )


def dry_run_report(models: list[str], datasets, modes: list[str]) -> None:
    print("Benchmark targets:")
    for dataset in datasets:
        print(f"  dataset={dataset.name}, images={len(dataset.samples)}")

    print("Model artifacts:")
    for model_name in models:
        spec = MODEL_SPECS[model_name]
        cfg = load_config(resolve_project_path(spec.config_path))
        if spec.kind == "yolo":
            best_path, last_path = yolo_weight_candidates(cfg)
            print(f"  {model_name}: best={best_path} exists={best_path.exists()} | last={last_path} exists={last_path.exists()}")
            continue
        checkpoint = cnn_checkpoint_path(cfg)
        svm_path = svm_model_path(cfg)
        for mode in modes:
            if mode == "cnn":
                print(f"  {model_name}: cnn={checkpoint} exists={checkpoint.exists()}")
            elif mode == "svm":
                print(f"  {model_name}: cnn={checkpoint} exists={checkpoint.exists()} | svm={svm_path} exists={svm_path.exists()}")


def write_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Model",
        "Dataset",
        "Status",
        "Images",
        "Mean ms/img",
        "Median ms/img",
        "Std ms/img",
        "FPS mean",
    ]
    table_rows = []
    for row in rows:
        model = f"{row['model']} - {row['mode']}"
        if row["status"] == "ok":
            std_ms = float(row["std_wall_time_s"]) * 1000.0 / max(int(row["num_images"]), 1)
            table_rows.append(
                {
                    "Model": model,
                    "Dataset": row["dataset"],
                    "Status": row["status"],
                    "Images": str(row["num_images"]),
                    "Mean ms/img": f"{float(row['mean_time_per_image_ms']):.3f}",
                    "Median ms/img": f"{float(row['median_time_per_image_ms']):.3f}",
                    "Std ms/img": f"{std_ms:.3f}",
                    "FPS mean": f"{float(row['images_per_second_mean']):.2f}",
                }
            )
        else:
            table_rows.append(
                {
                    "Model": model,
                    "Dataset": row["dataset"],
                    "Status": row["status"],
                    "Images": str(row["num_images"]),
                    "Mean ms/img": "",
                    "Median ms/img": "",
                    "Std ms/img": "",
                    "FPS mean": "",
                }
            )

    widths = {header: max(len(header), *(len(r[header]) for r in table_rows)) for header in headers}
    lines = [
        "Strict inference time benchmark",
        "",
        "Times are end-to-end dataloader inference wall times. Model loading is excluded.",
        "",
        " | ".join(header.ljust(widths[header]) for header in headers),
        " | ".join("-" * widths[header] for header in headers),
    ]
    for row in table_rows:
        lines.append(" | ".join(row[header].ljust(widths[header]) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    device = get_device(args.device)
    configure_device(device)
    print(f"Device: {describe_device(device)}")
    print(f"Warm-up runs: {args.warmup_runs}; measured runs: {args.runs}")

    rows: list[dict[str, Any]] = []
    for model_name in models:
        spec = MODEL_SPECS[model_name]
        cfg = load_config(resolve_project_path(spec.config_path))
        if spec.kind == "yolo":
            benchmark_yolo(spec, cfg, datasets, device, args, rows)
        else:
            benchmark_cnn_svm(spec, cfg, datasets, modes, device, args, rows)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.output_file
    txt_path = output_dir / args.txt_file

    fieldnames = [
        "model",
        "mode",
        "dataset",
        "status",
        "num_images",
        "batch_size",
        "num_workers",
        "warmup_runs",
        "measured_runs",
        "mean_wall_time_s",
        "median_wall_time_s",
        "std_wall_time_s",
        "min_wall_time_s",
        "max_wall_time_s",
        "mean_time_per_image_ms",
        "median_time_per_image_ms",
        "min_time_per_image_ms",
        "max_time_per_image_ms",
        "images_per_second_mean",
        "per_run_wall_time_s",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_txt(txt_path, rows)
    print(f"Saved benchmark CSV to: {csv_path}")
    print(f"Saved benchmark TXT to: {txt_path}")


if __name__ == "__main__":
    main()
