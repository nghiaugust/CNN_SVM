from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config import label_names, load_config
from src.data import LetterboxResize
from src.utils import set_seed


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    rel_path: Path
    label: int
    source: Path


@dataclass(frozen=True)
class AugmentOptions:
    input_size: tuple[int, int]
    pad_color: int
    copies_per_image: int
    flip_prob: float
    geometry_prob: float
    color_prob: float
    blur_prob: float
    noise_prob: float
    cutout_prob: float
    distortion_prob: float
    morphology_prob: float
    jpeg_quality: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a preprocessed and optionally augmented dataset from annotation text files."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root", default=None, help="Override dataset.root from config.")
    parser.add_argument("--output-root", default="dataset_preprocessed_augmented")
    parser.add_argument(
        "--augment-splits",
        default="train",
        help="Comma-separated splits to augment. Default: train. Use empty string for preprocessing only.",
    )
    parser.add_argument("--copies-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-original", action="store_true", help="Do not save preprocessed original images.")
    parser.add_argument("--preprocess-only", action="store_true", help="Disable all augmentation copies.")
    parser.add_argument("--flip-prob", type=float, default=0.0, help="Horizontal flip probability. Default is 0 for text.")
    parser.add_argument("--geometry-prob", type=float, default=0.85)
    parser.add_argument("--color-prob", type=float, default=0.70)
    parser.add_argument("--blur-prob", type=float, default=0.20)
    parser.add_argument("--noise-prob", type=float, default=0.25)
    parser.add_argument("--cutout-prob", type=float, default=0.20)
    parser.add_argument("--distortion-prob", type=float, default=0.25)
    parser.add_argument("--morphology-prob", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def parse_splits(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def read_annotation(root: Path, annotation_file: str | Path, names: list[str]) -> list[Sample]:
    annotation_path = root / annotation_file
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    samples: list[Sample] = []
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
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension at {annotation_path}:{line_no}: {source}")
        samples.append(Sample(rel_path=rel_path, label=label, source=source))
    return samples


def resolve_dataset_root(config_root: str | Path, override_root: str | Path | None) -> Path:
    if override_root is not None:
        return Path(override_root)

    root = Path(config_root)
    if root.exists():
        return root

    data_root = PROJECT_ROOT / "data" / root.name
    if data_root.exists():
        return data_root

    return root


def output_relative_path(sample: Sample, names: list[str], suffix: str = "") -> Path:
    class_name = names[sample.label]
    original_name = sample.rel_path.name
    stem = Path(original_name).stem
    ext = Path(original_name).suffix.lower() or ".jpg"
    filename = f"{stem}{suffix}{ext}"
    return Path(class_name) / filename


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(path, quality=quality, optimize=True)
    else:
        image.save(path)


def preprocess(image: Image.Image, options: AugmentOptions) -> Image.Image:
    fill = (options.pad_color, options.pad_color, options.pad_color)
    return LetterboxResize(options.input_size, fill=fill)(image)


def chance(rng: random.Random, probability: float) -> bool:
    return rng.random() < max(0.0, min(1.0, probability))


def random_geometry(image: Image.Image, rng: random.Random, options: AugmentOptions) -> Image.Image:
    fill = [options.pad_color, options.pad_color, options.pad_color]
    angle = rng.uniform(-3.0, 3.0)
    max_dx = max(1, int(image.width * 0.03))
    max_dy = max(1, int(image.height * 0.08))
    translate = (rng.randint(-max_dx, max_dx), rng.randint(-max_dy, max_dy))
    scale = rng.uniform(0.94, 1.06)
    shear = [rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0)]
    image = TF.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=InterpolationMode.BICUBIC,
        fill=fill,
    )
    if chance(rng, options.flip_prob):
        image = ImageOps.mirror(image)
    return image


def random_color(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.80, 1.25))
    if rng.random() < 0.25:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.75, 1.35))
    return image


def random_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.90)))


def random_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    sigma = rng.uniform(3.0, 10.0)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0.0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def random_cutout(image: Image.Image, rng: random.Random, options: AugmentOptions) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    holes = rng.randint(1, 2)
    fill = (options.pad_color, options.pad_color, options.pad_color)
    for _ in range(holes):
        width = rng.randint(max(4, image.width // 28), max(5, image.width // 10))
        height = rng.randint(max(4, image.height // 8), max(5, image.height // 3))
        left = rng.randint(0, max(0, image.width - width))
        top = rng.randint(0, max(0, image.height - height))
        draw.rectangle((left, top, left + width, top + height), fill=fill)
    return image


def grid_distortion(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    cols = 4
    rows = 2
    max_x = width * 0.025
    max_y = height * 0.060
    mesh = []
    for row in range(rows):
        for col in range(cols):
            x0 = int(col * width / cols)
            y0 = int(row * height / rows)
            x1 = int((col + 1) * width / cols)
            y1 = int((row + 1) * height / rows)
            bbox = (x0, y0, x1, y1)
            quad = (
                x0 + rng.uniform(-max_x, max_x),
                y0 + rng.uniform(-max_y, max_y),
                x1 + rng.uniform(-max_x, max_x),
                y0 + rng.uniform(-max_y, max_y),
                x1 + rng.uniform(-max_x, max_x),
                y1 + rng.uniform(-max_y, max_y),
                x0 + rng.uniform(-max_x, max_x),
                y1 + rng.uniform(-max_y, max_y),
            )
            mesh.append((bbox, quad))
    return image.transform(image.size, Image.Transform.MESH, mesh, Image.Resampling.BICUBIC)


def elastic_distortion(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    amp_x = rng.uniform(1.5, 4.0)
    amp_y = rng.uniform(0.5, 2.0)
    wave_x = rng.uniform(42.0, 95.0)
    wave_y = rng.uniform(80.0, 160.0)
    phase_x = rng.uniform(0, 2 * np.pi)
    phase_y = rng.uniform(0, 2 * np.pi)
    source_x = xx + amp_x * np.sin((yy / wave_x) * 2 * np.pi + phase_x)
    source_y = yy + amp_y * np.sin((xx / wave_y) * 2 * np.pi + phase_y)
    source_x = np.clip(np.rint(source_x), 0, width - 1).astype(np.int32)
    source_y = np.clip(np.rint(source_y), 0, height - 1).astype(np.int32)
    warped = arr[source_y, source_x]
    return Image.fromarray(warped, mode="RGB")


def random_distortion(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        return grid_distortion(image, rng)
    return elastic_distortion(image, rng)


def random_morphology(image: Image.Image, rng: random.Random) -> Image.Image:
    # For dark text on a light background, MinFilter thickens strokes and MaxFilter thins them.
    if rng.random() < 0.5:
        return image.filter(ImageFilter.MinFilter(size=3))
    return image.filter(ImageFilter.MaxFilter(size=3))


def augment(image: Image.Image, rng: random.Random, options: AugmentOptions) -> Image.Image:
    image = image.copy()
    if chance(rng, options.geometry_prob):
        image = random_geometry(image, rng, options)
    if chance(rng, options.color_prob):
        image = random_color(image, rng)
    if chance(rng, options.blur_prob):
        image = random_blur(image, rng)
    if chance(rng, options.noise_prob):
        image = random_noise(image, rng)
    if chance(rng, options.cutout_prob):
        image = random_cutout(image, rng, options)
    if chance(rng, options.distortion_prob):
        image = random_distortion(image, rng)
    if chance(rng, options.morphology_prob):
        image = random_morphology(image, rng)
    return image


def prepare_output(output_root: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace files.")
    output_root.mkdir(parents=True, exist_ok=True)


def write_annotation(output_root: Path, annotation_file: str | Path, rows: list[tuple[Path, int]], dry_run: bool) -> None:
    if dry_run:
        return
    text = "\n".join(f"{rel_path.as_posix()} {label}" for rel_path, label in rows)
    if text:
        text += "\n"
    annotation_path = output_root / annotation_file
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(text, encoding="utf-8")


def build_options(cfg: dict, args: argparse.Namespace) -> AugmentOptions:
    ds_cfg = cfg["dataset"]
    input_size_raw: Iterable[int] = ds_cfg.get("input_size", [128, 512])
    input_size = tuple(int(v) for v in input_size_raw)
    if len(input_size) != 2:
        raise ValueError(f"dataset.input_size must have two values, got: {input_size_raw}")
    copies = 0 if args.preprocess_only else max(0, int(args.copies_per_image))
    return AugmentOptions(
        input_size=(input_size[0], input_size[1]),
        pad_color=int(ds_cfg.get("pad_color", 255)),
        copies_per_image=copies,
        flip_prob=float(args.flip_prob),
        geometry_prob=float(args.geometry_prob),
        color_prob=float(args.color_prob),
        blur_prob=float(args.blur_prob),
        noise_prob=float(args.noise_prob),
        cutout_prob=float(args.cutout_prob),
        distortion_prob=float(args.distortion_prob),
        morphology_prob=float(args.morphology_prob),
        jpeg_quality=max(1, min(100, int(args.jpeg_quality))),
    )


def process_split(
    split: str,
    samples: list[Sample],
    names: list[str],
    output_root: Path,
    annotation_file: str | Path,
    options: AugmentOptions,
    augment_split: bool,
    include_original: bool,
    dry_run: bool,
    rng: random.Random,
) -> dict:
    annotation_rows: list[tuple[Path, int]] = []
    counts: Counter = Counter()
    generated = 0

    for sample in samples:
        if dry_run:
            if include_original:
                counts[sample.label] += 1
                annotation_rows.append((output_relative_path(sample, names), sample.label))
            if augment_split:
                counts[sample.label] += options.copies_per_image
                generated += options.copies_per_image
                for copy_idx in range(1, options.copies_per_image + 1):
                    rel_path = output_relative_path(sample, names, suffix=f"_aug{copy_idx:02d}")
                    annotation_rows.append((rel_path, sample.label))
            continue

        image = load_rgb(sample.source)
        base = preprocess(image, options)

        if include_original:
            rel_path = output_relative_path(sample, names)
            annotation_rows.append((rel_path, sample.label))
            counts[sample.label] += 1
            if not dry_run:
                save_image(base, output_root / rel_path, quality=options.jpeg_quality)

        if augment_split:
            for copy_idx in range(1, options.copies_per_image + 1):
                aug = augment(base, rng, options)
                rel_path = output_relative_path(sample, names, suffix=f"_aug{copy_idx:02d}")
                annotation_rows.append((rel_path, sample.label))
                counts[sample.label] += 1
                generated += 1
                if not dry_run:
                    save_image(aug, output_root / rel_path, quality=options.jpeg_quality)

    write_annotation(output_root, annotation_file, annotation_rows, dry_run=dry_run)
    return {
        "split": split,
        "source_images": len(samples),
        "saved_images": len(annotation_rows),
        "augmented_images": generated,
        "class_counts": {names[label]: int(counts.get(label, 0)) for label in range(len(names))},
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    rng = random.Random(seed)

    names = label_names(cfg)
    ds_cfg = cfg["dataset"]
    dataset_root = resolve_dataset_root(ds_cfg["root"], args.dataset_root)
    output_root = Path(args.output_root)
    augment_splits = parse_splits(args.augment_splits)
    options = build_options(cfg, args)

    prepare_output(output_root, overwrite=args.overwrite, dry_run=args.dry_run)

    summary: dict[str, object] = {
        "source_root": str(dataset_root),
        "output_root": str(output_root),
        "input_size": list(options.input_size),
        "pad_color": options.pad_color,
        "seed": seed,
        "augment_splits": sorted(augment_splits),
        "copies_per_image": options.copies_per_image,
        "include_original": not args.no_original,
        "splits": [],
    }

    total_by_class: defaultdict[str, int] = defaultdict(int)
    for split in ("train", "val", "test"):
        annotation_key = f"{split}_annotation"
        samples = read_annotation(dataset_root, ds_cfg[annotation_key], names)
        split_summary = process_split(
            split=split,
            samples=samples,
            names=names,
            output_root=output_root,
            annotation_file=ds_cfg[annotation_key],
            options=options,
            augment_split=split in augment_splits and options.copies_per_image > 0,
            include_original=not args.no_original,
            dry_run=args.dry_run,
            rng=rng,
        )
        summary["splits"].append(split_summary)  # type: ignore[index]
        for name, value in split_summary["class_counts"].items():
            total_by_class[name] += int(value)

    summary["total_class_counts"] = dict(total_by_class)
    if not args.dry_run:
        (output_root / "preprocess_augment_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.dry_run:
        print(f"Saved preprocessed dataset to: {output_root}")
        print("Update config dataset.root to this output root when you want to train on it.")


if __name__ == "__main__":
    main()
