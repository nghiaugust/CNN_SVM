from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config import load_config, save_config


TRAIN_DIR = PROJECT_ROOT / "train"

MODEL_CONFIGS = {
    "resnet18": Path("configs/config.yaml"),
    "resnet50": Path("configs/config_resnet50.yaml"),
    "convnext_tiny": Path("configs/config_convnext_tiny.yaml"),
    "deit_small": Path("configs/config_deit_small.yaml"),
    "yolov8": Path("configs/config_yolov8.yaml"),
}

CNN_SVM_MODELS = ("resnet18", "resnet50", "convnext_tiny", "deit_small")
YOLO_MODELS = ("yolov8",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all configured models sequentially.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or all. Supported: resnet18,resnet50,convnext_tiny,deit_small,yolov8.",
    )
    parser.add_argument(
        "--dataset-root",
        default="auto",
        help="Dataset root for annotation files. Default auto uses config root, then data/dataset fallback.",
    )
    parser.add_argument("--device", default="auto", help="Device passed to train scripts: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--config-output-dir", default="runs/train_all_configs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models whose expected artifacts already exist.")
    parser.add_argument("--skip-cnn", action="store_true", help="For CNN+SVM models, skip CNN and train SVM only.")
    parser.add_argument("--force-extract", action="store_true", help="Force SVM feature extraction.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with the next model if one training command fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and expected outputs without training.")
    parser.add_argument("--cnn-epochs", type=int, default=None, help="Override epochs for CNN backbones.")
    parser.add_argument("--yolo-epochs", type=int, default=None, help="Override epochs for YOLOv8.")
    parser.add_argument("--cnn-batch-size", type=int, default=None, help="Override batch size for CNN backbones.")
    parser.add_argument("--yolo-batch-size", type=int, default=None, help="Override batch size for YOLOv8.")
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


def validate_dataset_root(root: Path, cfg: dict[str, Any]) -> None:
    ds_cfg = cfg["dataset"]
    for key in ("train_annotation", "val_annotation", "test_annotation"):
        path = root / ds_cfg[key]
        if not path.exists():
            raise FileNotFoundError(f"Missing {key}: {path}")


def apply_common_overrides(cfg: dict[str, Any], model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = resolve_dataset_root(cfg["dataset"]["root"], args.dataset_root)
    validate_dataset_root(dataset_root, cfg)
    cfg["dataset"]["root"] = relative_to_project(dataset_root)

    if model_name in CNN_SVM_MODELS:
        if args.cnn_epochs is not None:
            cfg["training"]["epochs"] = int(args.cnn_epochs)
        if args.cnn_batch_size is not None:
            cfg["training"]["batch_size"] = int(args.cnn_batch_size)
    elif model_name in YOLO_MODELS:
        if args.yolo_epochs is not None:
            cfg["training"]["epochs"] = int(args.yolo_epochs)
        if args.yolo_batch_size is not None:
            cfg["training"]["batch_size"] = int(args.yolo_batch_size)

    return cfg


def write_run_config(model_name: str, args: argparse.Namespace) -> Path:
    cfg = load_config(project_path(MODEL_CONFIGS[model_name]))
    cfg = apply_common_overrides(cfg, model_name, args)
    config_dir = project_path(args.config_output_dir)
    config_path = config_dir / f"{model_name}.yaml"
    save_config(cfg, config_path)
    return config_path


def cnn_artifacts(cfg: dict[str, Any]) -> tuple[Path, Path]:
    cnn_checkpoint = project_path(Path(cfg["training"]["output_dir"]) / "best_cnn.pt")
    svm_model = project_path(Path(cfg["svm"]["output_dir"]) / "svm_model.joblib")
    return cnn_checkpoint, svm_model


def yolo_artifacts(cfg: dict[str, Any]) -> tuple[Path, Path]:
    run_dir = project_path(Path(cfg["training"]["output_dir"]) / str(cfg["training"]["run_name"]))
    return run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"


def expected_artifact_text(model_name: str, config_path: Path) -> str:
    cfg = load_config(config_path)
    if model_name in CNN_SVM_MODELS:
        cnn_checkpoint, svm_model = cnn_artifacts(cfg)
        return f"CNN={cnn_checkpoint} | SVM={svm_model}"
    best, last = yolo_artifacts(cfg)
    return f"YOLO best={best} | last={last}"


def should_skip_existing(model_name: str, config_path: Path, args: argparse.Namespace) -> bool:
    if not args.skip_existing:
        return False

    cfg = load_config(config_path)
    if model_name in CNN_SVM_MODELS:
        cnn_checkpoint, svm_model = cnn_artifacts(cfg)
        return cnn_checkpoint.exists() and svm_model.exists()

    best, last = yolo_artifacts(cfg)
    return best.exists() or last.exists()


def cnn_exists_for_skip(model_name: str, config_path: Path) -> bool:
    if model_name not in CNN_SVM_MODELS:
        return False
    cfg = load_config(config_path)
    cnn_checkpoint, _ = cnn_artifacts(cfg)
    return cnn_checkpoint.exists()


def command_for_model(model_name: str, config_path: Path, args: argparse.Namespace) -> list[str]:
    config_arg = relative_to_project(config_path)

    if model_name in CNN_SVM_MODELS:
        cmd = [sys.executable, str(TRAIN_DIR / "run_pipeline.py"), "--config", config_arg]
        if args.device is not None:
            cmd.extend(["--device", args.device])
        if args.skip_cnn or (args.skip_existing and cnn_exists_for_skip(model_name, config_path)):
            cmd.append("--skip-cnn")
        if args.force_extract:
            cmd.append("--force-extract")
        return cmd

    cmd = [sys.executable, str(TRAIN_DIR / "train_yolov8.py"), "--config", config_arg]
    if args.device is not None:
        cmd.extend(["--device", args.device])
    return cmd


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    failures: list[tuple[str, str]] = []
    start = time.perf_counter()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Models: {', '.join(models)}")

    for index, model_name in enumerate(models, start=1):
        print(f"\n[{index}/{len(models)}] Prepare {model_name}")
        config_path = write_run_config(model_name, args)
        print(f"Config: {config_path}")
        print(f"Expected artifacts: {expected_artifact_text(model_name, config_path)}")

        if should_skip_existing(model_name, config_path, args):
            print(f"Skip {model_name}: expected artifacts already exist.")
            continue

        cmd = command_for_model(model_name, config_path, args)
        try:
            run_command(cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            message = f"exit_code={exc.returncode}"
            failures.append((model_name, message))
            print(f"[ERROR] {model_name} failed: {message}", flush=True)
            if not args.continue_on_error:
                raise

    elapsed = time.perf_counter() - start
    print(f"\nFinished train_all_models in {elapsed / 60.0:.2f} minutes.")
    if failures:
        print("Failures:")
        for model_name, message in failures:
            print(f"  {model_name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
