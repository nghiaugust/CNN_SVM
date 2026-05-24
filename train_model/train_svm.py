from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import deep_update, label_names, load_config, save_config
from src.data import make_eval_loader
from src.features import extract_features
from src.metrics import save_classification_outputs
from src.models import infer_model_name, load_cnn_checkpoint
from src.utils import configure_device, describe_device, ensure_dir, get_device, save_json, set_seed
from src.visualize import (
    plot_binary_curves,
    plot_confusion,
    plot_misclassified_grid,
    plot_svm_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CNN features and train SVM with Grid Search.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cnn-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="Device to use for CNN feature extraction: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def extract_or_load_feature_cache(cfg: dict, checkpoint_path: str | Path, cache_path: str | Path, force: bool):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    device = get_device(str(cfg.get("device", "auto")))
    configure_device(device)
    print(f"Feature extraction device: {describe_device(device)}")
    names = label_names(cfg)
    model, checkpoint = load_cnn_checkpoint(checkpoint_path, num_classes=len(names), device=device)
    model_name = checkpoint.get("model_name", infer_model_name(model)) if isinstance(checkpoint, dict) else infer_model_name(model)

    batch_size = int(cfg["training"]["batch_size"])
    features = {}
    for split in ("train", "val", "test"):
        loader = make_eval_loader(cfg, split, batch_size=batch_size)
        X, y, paths = extract_features(
            model,
            loader,
            device=device,
            amp=bool(cfg["training"].get("amp", True)),
            desc=f"extract {split}",
        )
        features[f"X_{split}"] = X
        features[f"y_{split}"] = y
        features[f"paths_{split}"] = paths
    features["model_name"] = np.asarray(model_name)
    features["feature_dim"] = np.asarray(features["X_train"].shape[1], dtype="int64")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **features)
    return features


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = deep_update(
        cfg,
        {
            "device": args.device,
            "svm": {
                "output_dir": args.output_dir,
                "cnn_checkpoint": args.cnn_checkpoint,
            }
        },
    )
    if args.output_dir is not None:
        cfg["svm"]["feature_cache"] = str(Path(args.output_dir) / "features.npz")
    set_seed(int(cfg.get("seed", 42)))

    output_dir = ensure_dir(cfg["svm"]["output_dir"])
    save_config(cfg, output_dir / "config_used.yaml")
    names = label_names(cfg)
    checkpoint_path = Path(cfg["svm"]["cnn_checkpoint"])
    cache_path = Path(cfg["svm"]["feature_cache"])

    print(f"CNN checkpoint: {checkpoint_path}")
    print(f"Feature cache: {cache_path}")
    features = extract_or_load_feature_cache(cfg, checkpoint_path, cache_path, force=args.force_extract)

    X_trainval = np.vstack([features["X_train"], features["X_val"]])
    y_trainval = np.concatenate([features["y_train"], features["y_val"]])
    X_test = features["X_test"]
    y_test = features["y_test"]
    test_paths = features["paths_test"].astype(str).tolist()

    min_class = int(np.bincount(y_trainval).min())
    n_splits = max(2, min(int(cfg["svm"].get("cv", 5)), min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get("seed", 42)))

    svm_cfg = cfg["svm"]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=str(svm_cfg.get("kernel", "rbf")),
                    class_weight=svm_cfg.get("class_weight", "balanced"),
                    probability=bool(svm_cfg.get("probability", True)),
                ),
            ),
        ]
    )
    param_grid = {
        "svc__C": svm_cfg["C"],
        "svc__gamma": svm_cfg["gamma"],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=str(svm_cfg.get("scoring", "f1_macro")),
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2,
        return_train_score=True,
    )

    print(f"Train SVM on train+val features: X={X_trainval.shape}, cv={n_splits}")
    grid.fit(X_trainval, y_trainval)
    grid_results = pd.DataFrame(grid.cv_results_)
    grid_results.to_csv(output_dir / "grid_search_results.csv", index=False)
    plot_svm_grid(output_dir / "grid_search_results.csv", output_dir / "svm_grid_heatmap.png")

    best_params = {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "scoring": svm_cfg.get("scoring", "f1_macro"),
        "feature_dim": int(X_trainval.shape[1]),
        "trainval_samples": int(X_trainval.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }
    save_json(best_params, output_dir / "best_params.json")
    joblib.dump(grid.best_estimator_, output_dir / "svm_model.joblib")

    y_pred = grid.predict(X_test)
    probabilities = grid.predict_proba(X_test) if hasattr(grid, "predict_proba") else None
    metrics = save_classification_outputs(
        y_test,
        y_pred,
        names,
        output_dir,
        "svm_test",
        paths=test_paths,
        probabilities=probabilities,
    )
    plot_confusion(y_test, y_pred, names, output_dir / "svm_test_confusion.png", "SVM test")
    plot_binary_curves(y_test, probabilities, output_dir, "svm_test", positive_label=1)
    plot_misclassified_grid(test_paths, y_test, y_pred, names, output_dir / "svm_test_misclassified.png")

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV {svm_cfg.get('scoring', 'f1_macro')}: {grid.best_score_:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}, test macro F1: {metrics['f1_macro']:.4f}")
    print(f"Saved SVM artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
