from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from src.config import label_names, load_config
from src.data import NameAnnotationDataset
from src.utils import ensure_dir, set_seed
from src.visualize import make_sample_grid, plot_class_distribution, plot_size_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset overview plots.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="runs/dataset_overview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(args.output_dir)
    ds_cfg = cfg["dataset"]
    names = label_names(cfg)

    datasets = {
        "train": NameAnnotationDataset(ds_cfg["root"], ds_cfg["train_annotation"]),
        "val": NameAnnotationDataset(ds_cfg["root"], ds_cfg["val_annotation"]),
        "test": NameAnnotationDataset(ds_cfg["root"], ds_cfg["test_annotation"]),
    }
    plot_class_distribution(
        {split: dataset.class_counts() for split, dataset in datasets.items()},
        names,
        output_dir / "class_distribution.png",
    )

    size_rows = []
    all_paths = []
    for split, dataset in datasets.items():
        for image_path, label in dataset.samples:
            with Image.open(image_path) as image:
                width, height = image.size
            size_rows.append(
                {
                    "split": split,
                    "label": label,
                    "width": width,
                    "height": height,
                    "ratio": width / height,
                }
            )
            all_paths.append(str(image_path))
    plot_size_distribution(size_rows, output_dir / "image_size_distribution.png")
    random.shuffle(all_paths)
    make_sample_grid(all_paths, output_dir / "sample_grid.png", max_images=24)

    print(f"Saved dataset overview to: {Path(output_dir)}")


if __name__ == "__main__":
    main()
