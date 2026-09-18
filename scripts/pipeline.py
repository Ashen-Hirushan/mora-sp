"""
pipeline.py
-----------------------
Shared inference pipeline used by denoise.py (submission) and train.py
(validation during training), so that exactly the same code path is
measured in both.

    noisy --> classical.preprocess --> [noisy | cleaned | sigma] --> NAFNet --> output
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from classical import preprocess
from model import NAFNet

SIGMA_SCALE = 5.0   # sigma map (~0.03-0.17) is rescaled to ~O(1) before the network


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(img: np.ndarray, path: Path) -> None:
    # round (not truncate) when quantising -- truncation adds a -0.5 grey-level bias
    out = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)   # uint8 HxWx3 -> RGB PNG


def build_network_input(noisy: np.ndarray, cleaned: np.ndarray, sigma: np.ndarray) -> torch.Tensor:
    """Stack the 7 input channels into a (1, 7, H, W) float tensor."""
    stack = np.concatenate([noisy, cleaned, sigma[..., None] * SIGMA_SCALE], axis=2)
    return torch.from_numpy(np.ascontiguousarray(stack.transpose(2, 0, 1)))[None]


def load_network(weights: Path, device: torch.device) -> NAFNet:
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    model = NAFNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# Geometric self-ensemble: (flip_h, rot90 k) pairs, applied and exactly inverted.
_TTA_TRANSFORMS = [(False, 0), (True, 0), (False, 2), (True, 2),
                   (False, 1), (True, 1), (False, 3), (True, 3)]


@torch.inference_mode()
def run_network(model: NAFNet, x: torch.Tensor, tta: int = 1) -> torch.Tensor:
    """x: (1, 7, H, W) on the model's device. Returns (1, 3, H, W).
    tta = number of flip/rotation variants averaged (1, 2, 4 or 8)."""
    use_amp = x.device.type == "cuda"
    acc = None
    for flip, k in _TTA_TRANSFORMS[:tta]:
        t = torch.flip(x, dims=[3]) if flip else x
        t = torch.rot90(t, k, dims=[2, 3])
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            y = model(t).float()
        y = torch.rot90(y, -k, dims=[2, 3])
        y = torch.flip(y, dims=[3]) if flip else y
        acc = y if acc is None else acc + y
    return acc / tta


def denoise_image(noisy: np.ndarray, model: NAFNet | None, device: torch.device,
                  tta: int = 1) -> np.ndarray:
    """Full pipeline for one float32 HxWx3 image in [0, 1]."""
    pre = preprocess(noisy)
    x = build_network_input(noisy, pre["cleaned"], pre["sigma"]).to(device)
    y = run_network(model, x, tta)
    return y[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
