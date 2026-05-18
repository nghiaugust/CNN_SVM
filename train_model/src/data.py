from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import cpu_count_for_loader


class LetterboxResize:
    """Resize with aspect ratio preserved, then pad to a fixed canvas."""

    def __init__(self, size: Iterable[int], fill: int | tuple[int, int, int] = 255) -> None:
        height, width = list(size)
        self.height = int(height)
        self.width = int(width)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        src_w, src_h = image.size
        scale = min(self.width / src_w, self.height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        canvas = Image.new("RGB", (self.width, self.height), color=self.fill)
        left = (self.width - new_w) // 2
        top = (self.height - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas


class NameAnnotationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.annotation_file = self.root / annotation_file
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        for line_no, raw in enumerate(self.annotation_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid annotation at {self.annotation_file}:{line_no}: {raw}")
            rel_path, label_text = parts
            image_path = self.root / rel_path
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {self.annotation_file}:{line_no}: {image_path}")
            self.samples.append((image_path, int(label_text)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), str(image_path)

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(label for _, label in self.samples))

    def paths(self) -> list[str]:
        return [str(path) for path, _ in self.samples]

    def labels(self) -> list[int]:
        return [label for _, label in self.samples]


def build_transform(
    input_size: Iterable[int],
    train: bool,
    augment: bool,
    pad_color: int,
) -> transforms.Compose:
    fill = (pad_color, pad_color, pad_color)
    steps: list = [LetterboxResize(input_size, fill=fill)]
    if train and augment:
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


def make_dataset(cfg: dict, split: str, train_transform: bool = False) -> NameAnnotationDataset:
    ds_cfg = cfg["dataset"]
    annotation_key = f"{split}_annotation"
    transform = build_transform(
        input_size=ds_cfg["input_size"],
        train=train_transform,
        augment=bool(cfg["training"].get("augment", True)),
        pad_color=int(ds_cfg.get("pad_color", 255)),
    )
    return NameAnnotationDataset(ds_cfg["root"], ds_cfg[annotation_key], transform=transform)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    num_workers = cpu_count_for_loader(int(num_workers))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def make_cnn_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = make_dataset(cfg, "train", train_transform=True)
    val_ds = make_dataset(cfg, "val", train_transform=False)
    test_ds = make_dataset(cfg, "test", train_transform=False)
    batch_size = int(cfg["training"]["batch_size"])
    workers = int(cfg["dataset"].get("num_workers", 0))
    return (
        make_loader(train_ds, batch_size, shuffle=True, num_workers=workers),
        make_loader(val_ds, batch_size, shuffle=False, num_workers=workers),
        make_loader(test_ds, batch_size, shuffle=False, num_workers=workers),
    )


def make_eval_loader(cfg: dict, split: str, batch_size: int | None = None) -> DataLoader:
    dataset = make_dataset(cfg, split, train_transform=False)
    if batch_size is None:
        batch_size = int(cfg["training"]["batch_size"])
    return make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
    )
