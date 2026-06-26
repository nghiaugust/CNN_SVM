from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_all_models import (
    MODEL_SPECS,
    build_eval_datasets,
    configure_device,
    describe_device,
    dry_run_report,
    evaluate_cnn_svm_model,
    evaluate_yolo_model,
    get_device,
    load_config,
    parse_csv_list,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained CNN/SVM/YOLO models on the full data/evaluation set only."
    )
    parser.add_argument("--models", default="all", help="Comma-separated models or all.")
    parser.add_argument("--cnn-modes", default="cnn,svm", help="For CNN backbones: cnn,svm or both comma-separated.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--positive-label", type=int, default=0, help="Positive class for precision/recall/f1. Default 0 = Gach_Ten.")
    parser.add_argument("--output-dir", default="runs/evaluation_only")
    parser.add_argument("--metrics-file", default="metrics.csv")
    parser.add_argument("--predictions-file", default="predictions.csv")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only list the evaluation dataset and expected weight paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_csv_list(args.models, MODEL_SPECS.keys())
    modes = parse_csv_list(args.cnn_modes, {"cnn", "svm"})

    base_cfg = load_config(resolve_project_path("configs/config.yaml"))
    datasets = build_eval_datasets(base_cfg, ["evaluation"])

    if args.dry_run:
        dry_run_report(models, datasets, modes)
        return

    device = get_device(args.device)
    configure_device(device)
    print(f"Device: {describe_device(device)}")

    output_dir = resolve_project_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
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
