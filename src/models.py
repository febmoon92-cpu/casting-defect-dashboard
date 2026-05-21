"""Model definitions: baseline CNN (from scratch) and ResNet18 transfer learning."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Lightweight Conv-BN-ReLU x3 -> GAP -> FC."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = 2) -> nn.Module:
    name = name.lower()
    if name in {"baseline", "baseline_cnn", "cnn"}:
        return BaselineCNN(num_classes=num_classes)
    if name in {"resnet18", "resnet"}:
        return build_resnet18(num_classes=num_classes, pretrained=True)
    if name in {"efficientnet_b0", "effnet_b0", "effnet"}:
        return build_efficientnet_b0(num_classes=num_classes, pretrained=True)
    raise ValueError(f"Unknown model name: {name}")


def get_resnet_target_layer(model: nn.Module) -> nn.Module:
    """Return the last conv block of ResNet18, suitable for Grad-CAM."""
    return model.layer4[-1]


def get_target_layer_for(model: nn.Module) -> nn.Module:
    """Return a Grad-CAM-friendly final conv layer for any supported model."""
    if isinstance(model, BaselineCNN):
        return model.features[-3]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError(f"Unsupported model type for Grad-CAM: {type(model).__name__}")
