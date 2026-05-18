from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_history(history_csv: str | Path, output_path: str | Path) -> None:
    history = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["epoch"], history["train_f1_macro"], label="train")
    axes[2].plot(history["epoch"], history["val_f1_macro"], label="val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    _save(fig, output_path)


def plot_confusion(y_true, y_pred, label_names: list[str], output_path: str | Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    display.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
    ax.set_title(title)
    _save(fig, output_path)


def plot_binary_curves(y_true, probabilities, output_dir: str | Path, prefix: str, positive_label: int = 1) -> None:
    output_dir = Path(output_dir)
    if probabilities is None or probabilities.shape[1] != 2:
        return
    scores = probabilities[:, positive_label]

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=positive_label)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax)
    ax.set_title(f"{prefix} ROC")
    _save(fig, output_dir / f"{prefix}_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, scores, pos_label=positive_label)
    ap = average_precision_score(np.asarray(y_true) == positive_label, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap).plot(ax=ax)
    ax.set_title(f"{prefix} Precision-Recall")
    _save(fig, output_dir / f"{prefix}_pr_curve.png")


def plot_svm_grid(grid_results_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(grid_results_csv)
    pivot = df.pivot_table(
        index="param_svc__C",
        columns="param_svc__gamma",
        values="mean_test_score",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title("SVM Grid Search Mean CV Score")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, output_path)


def plot_class_distribution(counts_by_split: dict[str, dict[int, int]], label_names: list[str], output_path: str | Path) -> None:
    splits = list(counts_by_split.keys())
    labels = list(range(len(label_names)))
    x = np.arange(len(splits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, label in enumerate(labels):
        values = [counts_by_split[split].get(label, 0) for split in splits]
        ax.bar(x + (offset - 0.5) * width, values, width, label=label_names[label])
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Num images")
    ax.set_title("Class distribution by split")
    ax.legend()
    _save(fig, output_path)


def plot_size_distribution(size_rows: list[dict], output_path: str | Path) -> None:
    df = pd.DataFrame(size_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(df["width"], bins=30, color="#4C78A8")
    axes[0].set_title("Width")
    axes[1].hist(df["height"], bins=30, color="#F58518")
    axes[1].set_title("Height")
    axes[2].hist(df["ratio"], bins=30, color="#54A24B")
    axes[2].set_title("Width / Height")
    for ax in axes:
        ax.set_ylabel("Num images")
    _save(fig, output_path)


def make_sample_grid(image_paths: list[str], output_path: str | Path, max_images: int = 24) -> None:
    chosen = image_paths[:max_images]
    if not chosen:
        return
    thumb_w, thumb_h = 220, 72
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(chosen):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w - 8, thumb_h - 18), Image.Resampling.BICUBIC)
            x = (idx % cols) * thumb_w
            y = (idx // cols) * thumb_h
            canvas.paste(im, (x + 4, y + 4))
            draw.text((x + 4, y + thumb_h - 14), Path(path).as_posix()[-34:], fill=(40, 40, 40))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def plot_misclassified_grid(
    paths: list[str],
    y_true,
    y_pred,
    label_names: list[str],
    output_path: str | Path,
    max_images: int = 24,
) -> None:
    mistakes = [
        (path, int(t), int(p))
        for path, t, p in zip(paths, y_true, y_pred)
        if int(t) != int(p)
    ][:max_images]
    if not mistakes:
        return
    thumb_w, thumb_h = 260, 96
    cols = 3
    rows = int(np.ceil(len(mistakes) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (path, true_label, pred_label) in enumerate(mistakes):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w - 8, thumb_h - 28), Image.Resampling.BICUBIC)
            x = (idx % cols) * thumb_w
            y = (idx // cols) * thumb_h
            canvas.paste(im, (x + 4, y + 4))
            text = f"T:{label_names[true_label]}  P:{label_names[pred_label]}"
            draw.text((x + 4, y + thumb_h - 20), text, fill=(180, 30, 30))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
