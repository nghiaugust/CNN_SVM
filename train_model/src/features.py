from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .engine import _autocast
from .models import CNNFeatureExtractor


@torch.no_grad()
def extract_features(trained_cnn, loader, device: torch.device, amp: bool = True, desc: str = "features"):
    extractor = CNNFeatureExtractor(trained_cnn).to(device)
    extractor.eval()

    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    for images, batch_labels, batch_paths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with _autocast(device, amp):
            batch_features = extractor(images)
        features.append(batch_features.detach().cpu().numpy().astype("float32"))
        labels.extend(batch_labels.numpy().astype("int64").tolist())
        paths.extend(list(batch_paths))

    return np.vstack(features), np.asarray(labels, dtype="int64"), np.asarray(paths)
