from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, ResNet50_Weights, convnext_tiny, resnet18, resnet50


SUPPORTED_MODELS = {
    "resnet18": 512,
    "resnet50": 2048,
    "convnext_tiny": 768,
}


def normalize_model_name(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resnet_18": "resnet18",
        "resnet_50": "resnet50",
        "convnexttiny": "convnext_tiny",
        "convnext_t": "convnext_tiny",
        "convnext_tiny": "convnext_tiny",
    }
    return aliases.get(normalized, normalized)


def get_feature_dim(model_name: str) -> int:
    name = normalize_model_name(model_name)
    if name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
    return SUPPORTED_MODELS[name]


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.backbone_name = "resnet18"
    model.feature_dim = in_features
    return model


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.backbone_name = "resnet50"
    model.feature_dim = in_features
    return model


def build_convnext_tiny(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.backbone_name = "convnext_tiny"
    model.feature_dim = in_features
    return model


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = normalize_model_name(model_name)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    if name == "convnext_tiny":
        return build_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
    supported = ", ".join(sorted(SUPPORTED_MODELS))
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")


def infer_model_name(model: nn.Module) -> str:
    if hasattr(model, "backbone_name"):
        return normalize_model_name(str(model.backbone_name))
    if hasattr(model, "fc"):
        return "resnet18"
    if hasattr(model, "classifier") and hasattr(model, "features") and hasattr(model, "avgpool"):
        return "convnext_tiny"
    raise ValueError("Cannot infer CNN backbone name from model instance.")


class CNNFeatureExtractor(nn.Module):
    def __init__(self, trained_cnn: nn.Module, model_name: str | None = None) -> None:
        super().__init__()
        self.model_name = normalize_model_name(model_name or infer_model_name(trained_cnn))
        if self.model_name in {"resnet18", "resnet50"}:
            self.features = nn.Sequential(*list(trained_cnn.children())[:-1])
        elif self.model_name == "convnext_tiny":
            self.features = trained_cnn.features
            self.avgpool = trained_cnn.avgpool
            self.pre_classifier = nn.Sequential(*list(trained_cnn.classifier.children())[:-1])
        else:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            raise ValueError(f"Unsupported feature extractor '{self.model_name}'. Supported models: {supported}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == "convnext_tiny":
            x = self.features(x)
            x = self.avgpool(x)
            return self.pre_classifier(x)

        x = self.features(x)
        return torch.flatten(x, 1)


def load_cnn_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int = 2,
    device: torch.device | str = "cpu",
    model_name: str | None = None,
) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_section = checkpoint_cfg.get("model", {}) if isinstance(checkpoint_cfg, dict) else {}
    if not isinstance(model_section, dict):
        model_section = {}
    checkpoint_model_name = None
    if isinstance(checkpoint, dict):
        checkpoint_model_name = checkpoint.get("model_name") or model_section.get("name")
    resolved_model_name = model_name or checkpoint_model_name or "resnet18"
    model = build_model(resolved_model_name, num_classes=num_classes, pretrained=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, checkpoint
