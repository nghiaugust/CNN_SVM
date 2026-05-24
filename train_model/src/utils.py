from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    requested = str(name or "auto").strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "gpu":
        requested = "cuda"

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch install cannot use CUDA. "
                f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}, "
                f"cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}. "
                "Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ "
                "or set device to 'cpu'/'auto'."
            )
        device_count = torch.cuda.device_count()
        if device.index is not None and not 0 <= device.index < device_count:
            raise ValueError(f"CUDA device index {device.index} is invalid; found {device_count} CUDA device(s).")
        if device.index is not None:
            torch.cuda.set_device(device.index)
    return device


def configure_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def describe_device(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    memory_gb = props.total_memory / (1024**3)
    return f"{device} ({props.name}, {memory_gb:.1f} GB)"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cpu_count_for_loader(default: int) -> int:
    if os.name == "nt":
        return min(default, 2)
    return default
