"""Self-contained Grad-CAM utilities for ResNet18 / EfficientNet-B0 / BaselineCNN.

This module intentionally avoids the upstream ``pytorch_grad_cam`` (PyPI name
``grad-cam``) dependency, because it transitively requires ``opencv-python``
(non-headless), which in turn requires ``libGL`` / ``libSM`` / ``libXext`` and
other X11 system libraries that are awkward to install on minimal Streamlit
Community Cloud images.

The Grad-CAM logic implemented here is functionally equivalent for our use
cases:
  - a single target conv layer,
  - a single classification target per sample,
  - per-sample min-max normalization to ``[0, 1]``,
  - bilinear upsampling back to the input resolution.

We also re-export :class:`ClassifierOutputTarget` so call sites can keep their
existing import style (`from src.gradcam import GradCAM, ClassifierOutputTarget`).

Visualization (``show_cam_on_image``) uses ``matplotlib`` colormaps instead of
``cv2.applyColorMap``, removing the last reason to install OpenCV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import config
from .models import BaselineCNN, build_model, get_resnet_target_layer


# --------------------------------------------------------------------------- #
# pytorch_grad_cam API shims
# --------------------------------------------------------------------------- #


class ClassifierOutputTarget:
    """Drop-in replacement for ``pytorch_grad_cam.utils.model_targets``.

    Returns the scalar (per-sample) logit / score for ``category`` from a
    classification output of shape ``[B, C]`` or ``[C]``.
    """

    def __init__(self, category: int):
        self.category = int(category)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# --------------------------------------------------------------------------- #
# Grad-CAM core
# --------------------------------------------------------------------------- #


class GradCAM:
    """Lightweight Grad-CAM with the same high-level API as ``pytorch_grad_cam``.

    Usage::

        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])
        # grayscale.shape == (B, H, W), each plane normalized to [0, 1]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Sequence[torch.nn.Module],
    ) -> None:
        if len(target_layers) != 1:
            raise ValueError(
                "This lightweight GradCAM supports exactly one target layer; "
                f"got {len(target_layers)}."
            )
        self.model = model
        self.target_layer = target_layers[0]
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> "GradCAM":
        def fwd_hook(_m, _inp, output: torch.Tensor) -> None:
            self._activations = output

        def bwd_hook(_m, _grad_input, grad_output) -> None:
            self._gradients = grad_output[0]

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))
        return self

    def __exit__(self, *_exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._activations = None
        self._gradients = None

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: Sequence[ClassifierOutputTarget],
    ) -> np.ndarray:
        if not self._handles:
            raise RuntimeError("GradCAM must be used inside a `with` block.")
        if input_tensor.dim() != 4:
            raise ValueError(f"input_tensor must be 4D (B, C, H, W); got {tuple(input_tensor.shape)}")

        batch_size = input_tensor.shape[0]
        if len(targets) == 1 and batch_size > 1:
            targets = list(targets) * batch_size
        if len(targets) != batch_size:
            raise ValueError(
                f"len(targets)={len(targets)} must equal batch size={batch_size} (or be 1)."
            )

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(input_tensor)
            score = torch.stack(
                [t(logits[i : i + 1]).sum() for i, t in enumerate(targets)]
            ).sum()
            score.backward()

        activations = self._activations
        gradients = self._gradients
        if activations is None or gradients is None:
            raise RuntimeError(
                "Forward/backward hooks did not fire — verify that `target_layer` is "
                "actually traversed by the model on this input."
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)              # [B, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)          # [B, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze(1).detach().cpu().numpy().astype(np.float32)  # [B, H, W]

        for i in range(cam.shape[0]):
            mn, mx = float(cam[i].min()), float(cam[i].max())
            if mx > mn:
                cam[i] = (cam[i] - mn) / (mx - mn)
            else:
                cam[i] = np.zeros_like(cam[i])
        return cam


# --------------------------------------------------------------------------- #
# Visualization (matplotlib colormap; no cv2 dependency)
# --------------------------------------------------------------------------- #


def _resolve_colormap(name: str):
    """Return a matplotlib colormap by name, compatible with mpl >= 3.6."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    return matplotlib.cm.get_cmap(name)  # pragma: no cover - mpl < 3.6 fallback


def show_cam_on_image(
    rgb_in_01: np.ndarray,
    grayscale_cam: np.ndarray,
    use_rgb: bool = True,                # kept for API compat; we always emit RGB
    colormap_name: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto an RGB image.

    Args:
        rgb_in_01:     ``(H, W, 3)`` float, values in ``[0, 1]``.
        grayscale_cam: ``(H, W)`` float in ``[0, 1]``.
        colormap_name: any matplotlib colormap name (default ``"jet"``).
        alpha:         blend weight for the heatmap (0.5 ≈ even mix).

    Returns:
        ``(H, W, 3)`` ``uint8`` overlay image.
    """
    del use_rgb  # noqa: F841 - kept for backwards compatibility with the original API
    if rgb_in_01.ndim != 3 or rgb_in_01.shape[2] != 3:
        raise ValueError(f"rgb_in_01 must be (H, W, 3); got {rgb_in_01.shape}")
    if grayscale_cam.shape != rgb_in_01.shape[:2]:
        raise ValueError(
            f"grayscale_cam shape {grayscale_cam.shape} must match image spatial "
            f"shape {rgb_in_01.shape[:2]}."
        )
    cmap = _resolve_colormap(colormap_name)
    heatmap = cmap(np.clip(grayscale_cam, 0.0, 1.0))[..., :3]  # drop alpha channel
    overlay = alpha * heatmap + (1.0 - alpha) * np.clip(rgb_in_01, 0.0, 1.0)
    return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# High-level helpers (unchanged signatures)
# --------------------------------------------------------------------------- #


def _eval_transform(img_size: int = config.IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, BaselineCNN):
        return model.features[-3]
    return get_resnet_target_layer(model)


def load_model_for_cam(model_name: str, ckpt_path: Path, device: torch.device):
    model = build_model(model_name)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def overlay_for_image(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
    target_class: int = 1,
    img_size: int = config.IMG_SIZE,
) -> tuple[np.ndarray, float, int]:
    """Return ``(overlay_uint8, prob_defect, pred_class)`` for a single image."""
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
        grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
    overlay = show_cam_on_image(rgb_arr, grayscale, use_rgb=True)
    return overlay, float(probs[1]), pred


def batch_overlays(
    model: torch.nn.Module,
    image_paths: Iterable[str | Path],
    device: torch.device,
    target_class: int = 1,
) -> list[dict]:
    results: list[dict] = []
    for p in image_paths:
        try:
            overlay, prob, pred = overlay_for_image(model, p, device, target_class)
            results.append(
                {"path": str(p), "overlay": overlay, "prob_defect": prob, "pred": pred}
            )
        except Exception as e:  # pragma: no cover
            results.append({"path": str(p), "error": str(e)})
    return results
