from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import deep_update, label_names, load_config, save_config


LETTERBOX_SIZE: tuple[int, int] = (128, 512)
PAD_COLOR = 255
CUSTOM_AUGMENT = True


@dataclass(frozen=True)
class AnnotationSample:
    rel_path: Path
    label: int
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 classification model from annotation text files.")
    parser.add_argument("--config", default="config_yolov8.yaml")
    parser.add_argument("--model", default=None, help="Override YOLO checkpoint, e.g. yolov8s-cls.pt.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", default=None, help="Override YOLO imgsz, e.g. 512.")
    parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, cuda:0, 0, ...")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-prepare", action="store_true", help="Skip building dataset.yolo_root.")
    parser.add_argument("--prepare-only", action="store_true", help="Build dataset.yolo_root and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and annotations without writing files.")
    return parser.parse_args()


def parse_size(value: Any) -> int | list[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (list, tuple)):
        parts = [int(x) for x in value]
    else:
        text = str(value).lower().replace("x", ",")
        parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Invalid size value: {value!r}")


def train_imgsz(cfg: dict[str, Any]) -> int:
    parsed = parse_size(cfg["training"].get("imgsz", 224))
    if isinstance(parsed, list):
        return max(parsed)
    return int(parsed or 224)


def letterbox_size(cfg: dict[str, Any]) -> tuple[int, int]:
    raw = cfg["training"].get("letterbox_size") or cfg["dataset"].get("input_size") or cfg["training"].get("imgsz", 224)
    parsed = parse_size(raw)
    if isinstance(parsed, list):
        return int(parsed[0]), int(parsed[1])
    size = int(parsed or 224)
    return size, size


def yolo_device(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return None
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        return text.split(":", 1)[1]
    return value


def read_annotation(root: Path, annotation_file: str | Path, names: list[str]) -> list[AnnotationSample]:
    annotation_path = root / annotation_file
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    samples: list[AnnotationSample] = []
    for line_no, raw in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid annotation at {annotation_path}:{line_no}: {raw}")
        rel_path = Path(parts[0])
        label = int(parts[1])
        if label < 0 or label >= len(names):
            raise ValueError(f"Invalid label at {annotation_path}:{line_no}: {label}")
        source = root / rel_path
        if not source.exists():
            raise FileNotFoundError(f"Image not found at {annotation_path}:{line_no}: {source}")
        samples.append(AnnotationSample(rel_path=rel_path, label=label, source=source))
    return samples


def destination_relative_path(sample: AnnotationSample, names: list[str]) -> Path:
    parts = sample.rel_path.parts
    if parts and parts[0] == names[sample.label]:
        remaining = parts[1:]
        if remaining:
            return Path(*remaining)
    return Path(sample.rel_path.name)


def materialize_image(source: Path, destination: Path, mode: str) -> str:
    if destination.exists():
        return "existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copied"
    if mode == "symlink":
        try:
            os.symlink(source, destination)
            return "symlinked"
        except OSError:
            shutil.copy2(source, destination)
            return "copied"

    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def prepare_yolo_dataset(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Counter]:
    ds_cfg = cfg["dataset"]
    root = Path(ds_cfg["root"])
    yolo_root = Path(ds_cfg.get("yolo_root", "dataset_yolov8_cls"))
    names = label_names(cfg)
    link_mode = str(ds_cfg.get("link_mode", "hardlink")).lower()
    if link_mode not in {"hardlink", "copy", "symlink"}:
        raise ValueError("dataset.link_mode must be one of: hardlink, copy, symlink")

    if names != sorted(names):
        print(
            "WARNING: Ultralytics ImageFolder sorts class folders alphabetically. "
            f"Current label order is {names}, sorted order is {sorted(names)}."
        )

    summaries: dict[str, Counter] = {}
    actions: Counter = Counter()
    for split in ("train", "val", "test"):
        annotation_key = f"{split}_annotation"
        samples = read_annotation(root, ds_cfg[annotation_key], names)
        counts = Counter(sample.label for sample in samples)
        summaries[split] = counts

        if dry_run:
            continue

        for name in names:
            (yolo_root / split / name).mkdir(parents=True, exist_ok=True)

        for sample in samples:
            class_name = names[sample.label]
            dst_rel = destination_relative_path(sample, names)
            destination = yolo_root / split / class_name / dst_rel
            action = materialize_image(sample.source, destination, link_mode)
            actions[action] += 1

    if not dry_run:
        print("Prepared YOLO classification dataset:", yolo_root)
        print("File actions:", ", ".join(f"{key}={value}" for key, value in sorted(actions.items())) or "none")
    return summaries


def print_summary(summaries: dict[str, Counter], names: list[str]) -> None:
    for split in ("train", "val", "test"):
        counts = summaries.get(split, Counter())
        total = sum(counts.values())
        parts = ", ".join(f"{names[i]}={counts.get(i, 0)}" for i in range(len(names)))
        print(f"{split}: total={total}, {parts}")


def configure_letterbox(cfg: dict[str, Any]) -> None:
    global LETTERBOX_SIZE, PAD_COLOR, CUSTOM_AUGMENT
    LETTERBOX_SIZE = letterbox_size(cfg)
    PAD_COLOR = int(cfg["dataset"].get("pad_color", 255))
    CUSTOM_AUGMENT = bool(cfg["training"].get("augment", True))


def build_letterbox_transforms(augment: bool):
    from torchvision import transforms

    from src.data import LetterboxResize

    fill = (PAD_COLOR, PAD_COLOR, PAD_COLOR)
    steps: list[Any] = [LetterboxResize(LETTERBOX_SIZE, fill=fill)]
    if augment:
        steps.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.12, contrast=0.18)],
                    p=0.45,
                ),
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.02, 0.05),
                    scale=(0.96, 1.04),
                    shear=(-2, 2),
                    fill=fill,
                ),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


try:
    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
except ImportError:
    YOLO = None
    ClassificationDataset = None
    ClassificationTrainer = None
    ClassificationValidator = None


if ClassificationDataset is not None:

    class LetterboxClassificationDataset(ClassificationDataset):
        def __init__(self, root: str, args, augment: bool = False, prefix: str = "", apply_augment: bool | None = None):
            super().__init__(root=root, args=args, augment=augment, prefix=prefix)
            self.torch_transforms = build_letterbox_transforms(bool(augment if apply_augment is None else apply_augment))


    class LetterboxClassificationValidator(ClassificationValidator):
        def build_dataset(self, img_path: str) -> ClassificationDataset:
            return LetterboxClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)


    class LetterboxClassificationTrainer(ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch=None) -> ClassificationDataset:
            is_train = mode == "train"
            return LetterboxClassificationDataset(
                root=img_path,
                args=self.args,
                augment=is_train,
                prefix=mode,
                apply_augment=is_train and CUSTOM_AUGMENT,
            )

        def get_validator(self):
            self.loss_names = ["loss"]
            return LetterboxClassificationValidator(
                self.test_loader,
                self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )


def build_train_args(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg["training"]
    project = Path(train_cfg.get("output_dir", "runs/yolov8_cls"))
    run_name = str(train_cfg.get("run_name", Path(str(cfg["model"]["name"])).stem))
    args: dict[str, Any] = {
        "data": str(Path(cfg["dataset"].get("yolo_root", "dataset_yolov8_cls"))),
        "epochs": int(train_cfg["epochs"]),
        "batch": int(train_cfg["batch_size"]),
        "imgsz": train_imgsz(cfg),
        "project": str(project),
        "name": run_name,
        "workers": int(train_cfg.get("workers", cfg["dataset"].get("num_workers", 0))),
        "seed": int(cfg.get("seed", 42)),
        "pretrained": bool(cfg["model"].get("pretrained", True)),
        "exist_ok": bool(train_cfg.get("exist_ok", True)),
    }

    device = yolo_device(cfg.get("device", "auto"))
    if device is not None:
        args["device"] = device

    optional_keys = (
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "optimizer",
        "patience",
        "amp",
        "cache",
        "plots",
        "val",
        "save_period",
        "dropout",
        "deterministic",
    )
    for key in optional_keys:
        if key in train_cfg and train_cfg[key] is not None:
            args[key] = train_cfg[key]
    return args


def train_yolo(cfg: dict[str, Any]) -> None:
    if YOLO is None:
        raise SystemExit("Missing dependency 'ultralytics'. Install it with: pip install -r requirements.txt")

    configure_letterbox(cfg)
    train_args = build_train_args(cfg)
    project = Path(train_args["project"])
    run_name = str(train_args["name"])
    run_dir = project / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_used.yaml")

    model_name = str(cfg["model"]["name"])
    print(f"Model: {model_name}")
    print(f"Data: {train_args['data']}")
    print(f"Output: {run_dir}")
    if bool(cfg["training"].get("letterbox", True)):
        print(f"Letterbox size: {LETTERBOX_SIZE[0]}x{LETTERBOX_SIZE[1]}")

    model = YOLO(model_name)
    trainer_cls = LetterboxClassificationTrainer if bool(cfg["training"].get("letterbox", True)) else None
    model.train(trainer=trainer_cls, **train_args)

    eval_cfg = cfg.get("evaluation", {})
    if not bool(eval_cfg.get("enabled", True)):
        return

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        weights = run_dir / "weights" / "last.pt"
    if not weights.exists():
        print("No trained weights found for evaluation.")
        return

    eval_model = YOLO(str(weights))
    validator_cls = LetterboxClassificationValidator if bool(cfg["training"].get("letterbox", True)) else None
    for split in eval_cfg.get("splits", ["val", "test"]):
        print(f"Evaluating split: {split}")
        eval_model.val(
            validator=validator_cls,
            data=train_args["data"],
            split=split,
            imgsz=train_args["imgsz"],
            batch=train_args["batch"],
            workers=train_args["workers"],
            device=train_args.get("device", None),
            project=str(run_dir / "eval"),
            name=str(split),
            exist_ok=True,
            plots=bool(cfg["training"].get("plots", True)),
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {
        "device": args.device,
        "model": {"name": args.model},
        "training": {
            "output_dir": args.output_dir,
            "run_name": args.run_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "imgsz": parse_size(args.imgsz),
            "augment": False if args.no_augment else None,
        },
    }
    cfg = deep_update(cfg, overrides)

    names = label_names(cfg)
    if args.dry_run:
        summaries = prepare_yolo_dataset(cfg, dry_run=True)
        print_summary(summaries, names)
        print(f"YOLO dataset root: {cfg['dataset'].get('yolo_root', 'dataset_yolov8_cls')}")
        return

    if not args.no_prepare:
        summaries = prepare_yolo_dataset(cfg, dry_run=False)
        print_summary(summaries, names)

    if args.prepare_only:
        return

    train_yolo(cfg)


if __name__ == "__main__":
    main()
