from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import save_json, write_text


def classification_metrics(y_true, y_pred, label_names: list[str]) -> dict:
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_classification_outputs(
    y_true,
    y_pred,
    label_names: list[str],
    output_dir: str | Path,
    prefix: str,
    paths: list[str] | None = None,
    probabilities: np.ndarray | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(y_true, y_pred, label_names)
    save_json(metrics, output_dir / f"{prefix}_metrics.json")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    write_text(report_text, output_dir / f"{prefix}_classification_report.txt")

    rows = {
        "y_true": y_true,
        "y_pred": y_pred,
        "true_name": [label_names[int(v)] for v in y_true],
        "pred_name": [label_names[int(v)] for v in y_pred],
        "correct": [int(t) == int(p) for t, p in zip(y_true, y_pred)],
    }
    if paths is not None:
        rows["path"] = paths
    if probabilities is not None:
        for idx, name in enumerate(label_names):
            rows[f"prob_{name}"] = probabilities[:, idx]
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    return metrics
