"""Grad-CAM utilities for ResNet18 (and Baseline CNN)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from . import config
from .models import BaselineCNN, build_model, get_resnet_target_layer


def _eval_transform(img_size: int = config.IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_target_layer(model: torch.nn.Module):
    if isinstance(model, BaselineCNN):
        return model.features[-3]
    return get_resnet_target_layer(model)


def load_model_for_cam(model_name: str, ckpt_path: Path, device: torch.device):
    model = build_model(model_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    model.to(device).eval()
    return model


def overlay_for_image(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
    target_class: int = 1,
    img_size: int = config.IMG_SIZE,
) -> tuple[np.ndarray, float, int]:
    """Return (overlay_uint8, prob_defect, pred_class) for a single image path."""
    tf = _eval_transform(img_size)
    with Image.open(image_path) as raw:
        rgb = raw.convert("RGB").resize((img_size, img_size))
    rgb_arr = np.asarray(rgb).astype(np.float32) / 255.0

    x = tf(Image.fromarray((rgb_arr * 255).astype(np.uint8))).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())

    target_layer = get_target_layer(model)
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
    overlay = show_cam_on_image(rgb_arr, grayscale_cam, use_rgb=True)
    return overlay, float(probs[1]), pred


def batch_overlays(
    model: torch.nn.Module,
    image_paths: Iterable[str | Path],
    device: torch.device,
    target_class: int = 1,
) -> list[dict]:
    results = []
    for p in image_paths:
        try:
            overlay, prob, pred = overlay_for_image(model, p, device, target_class)
            results.append({"path": str(p), "overlay": overlay, "prob_defect": prob, "pred": pred})
        except Exception as e:
            results.append({"path": str(p), "error": str(e)})
    return results
