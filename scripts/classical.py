"""
classical.py
-----------------------
Classical signal-processing stages of the pipeline. All functions are blind:
they estimate everything they need (noise level, defect pixels, banding) from
the single input image, with no knowledge of any ground truth or masks.

Stages
    1. estimate_noise_model   -- Poisson-Gaussian fit  sigma^2(I) = a*I + b
    2. correct_defect_pixels  -- adaptive (sigma-scaled) median outlier repair
    3. remove_banding         -- row/column fixed-offset (stripe) removal
    4. classical_denoise      -- generalised Anscombe VST + Non-Local Means
                                 (used as the no-network fallback)

All images are float32 arrays in [0, 1] with shape (H, W, 3).
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Noise model estimation
# ---------------------------------------------------------------------------
def estimate_noise_model(img: np.ndarray, n_bins: int = 12) -> tuple[float, float]:
    """Fit the signal-dependent noise model sigma^2(I) = a*I + b.

    A Laplacian-like high-pass (Immerkaer 1996) isolates noise from image
    structure; pixels are binned by local intensity and a robust (MAD) noise
    estimate is taken per bin. Only the flattest pixels of each bin are used
    so edges do not inflate the estimate. A weighted least-squares line in
    (intensity, variance) space gives (a, b).
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    # Var of the filter output for white noise = 36 * sigma^2
    gray = img.mean(axis=2, dtype=np.float32)
    hp = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT) / 6.0
    local_mean = cv2.blur(img, (7, 7))
    # Structure measure: gradient energy of a smoothed image
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)

    flat = grad < np.percentile(grad, 50)
    inten = local_mean[flat].ravel()
    resid = hp[flat].ravel()

    edges = np.linspace(0.02, 0.95, n_bins + 1)
    xs, ys, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (inten >= lo) & (inten < hi)
        count = int(sel.sum())
        if count < 2000:
            continue
        r = resid[sel]
        sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
        xs.append(0.5 * (lo + hi))
        ys.append(sigma * sigma)
        ws.append(np.sqrt(count))

    if len(xs) < 2:
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        return 0.0, float(sigma * sigma)

    a, b = np.polyfit(np.array(xs), np.array(ys), 1, w=np.array(ws))
    a = max(float(a), 0.0)
    b = max(float(b), 1e-6)
    return a, b


def noise_sigma_map(img: np.ndarray, a: float, b: float) -> np.ndarray:
    """Per-pixel noise standard deviation predicted by the fitted model."""
    local = cv2.GaussianBlur(img.mean(axis=2, dtype=np.float32), (0, 0), 3.0)
    return np.sqrt(np.clip(a * np.clip(local, 0, 1) + b, 1e-6, None)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Defect (hot / dead) pixel correction
# ---------------------------------------------------------------------------
def correct_defect_pixels(img: np.ndarray, sigma_map: np.ndarray,
                          k: float = 5.0) -> np.ndarray:
    """Replace isolated outliers with their 3x3 median.

    Unlike a fixed threshold, the decision threshold scales with the local
    noise level (k * sigma), so heavy but legitimate shot noise in bright
    areas is not mistaken for a defect, while defects in dark areas are caught.
    A pixel must also deviate from *all* of its 8 neighbours in the same
    direction, which protects thin bright/dark structures such as lines.
    """
    out = img.copy()
    thr = k * sigma_map
    for c in range(3):
        ch = img[..., c]
        med = cv2.medianBlur(ch, 3)
        dev = ch - med
        cand = np.abs(dev) > thr
        if not cand.any():
            continue
        # isolated-peak test: strictly above max (or below min) of neighbours
        nb_kernel = np.ones((3, 3), np.uint8)
        nb_kernel[1, 1] = 0
        nb_max = cv2.dilate(ch, nb_kernel)
        nb_min = cv2.erode(ch, nb_kernel)
        isolated = (ch > nb_max) | (ch < nb_min)
        mask = cand & isolated
        out[..., c] = np.where(mask, med, ch)
    return out


# ---------------------------------------------------------------------------
# 3. Row / column banding removal
# ---------------------------------------------------------------------------
def _line_offsets(lum: np.ndarray, axis: int, smooth_sigma: float) -> np.ndarray:
    """Robust per-line offset along `axis` (0 -> rows, 1 -> columns).

    The image is smoothed *across* lines only; subtracting it removes image
    content that varies slowly across lines and keeps line-constant offsets.
    The median along each line rejects localised edges.
    """
    if axis == 0:   # offsets per row: smooth along y
        smoothed = cv2.GaussianBlur(lum, (1, 0), sigmaX=1e-3, sigmaY=smooth_sigma)
        resid = lum - smoothed
        return np.median(resid, axis=1)
    smoothed = cv2.GaussianBlur(lum, (0, 1), sigmaX=smooth_sigma, sigmaY=1e-3)
    resid = lum - smoothed
    return np.median(resid, axis=0)


def remove_banding(img: np.ndarray, iterations: int = 2,
                   smooth_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate and subtract row and column offsets (shared by all channels,
    as the offsets were observed to be ~0.97 correlated between channels)."""
    out = img.copy()
    total_row = np.zeros(img.shape[0], np.float32)
    total_col = np.zeros(img.shape[1], np.float32)
    for _ in range(iterations):
        lum = out.mean(axis=2, dtype=np.float32)
        row = _line_offsets(lum, 0, smooth_sigma).astype(np.float32)
        out -= row[:, None, None]
        total_row += row
        lum = out.mean(axis=2, dtype=np.float32)
        col = _line_offsets(lum, 1, smooth_sigma).astype(np.float32)
        out -= col[None, :, None]
        total_col += col
    # The median of the high-pass residual is biased by skewed (clipped,
    # signal-dependent) noise, so only the zero-mean *pattern* is trusted;
    # the global offset is left untouched.
    out += total_row.mean() + total_col.mean()
    return out, total_row - total_row.mean(), total_col - total_col.mean()


# ---------------------------------------------------------------------------
# Combined pre-processing used before the network
# ---------------------------------------------------------------------------
def preprocess(img: np.ndarray) -> dict:
    """Run stages 1-3. Returns the cleaned image, sigma map and model params."""
    a, b = estimate_noise_model(img)
    sigma = noise_sigma_map(img, a, b)
    cleaned = correct_defect_pixels(img, sigma)
    cleaned, _, _ = remove_banding(cleaned)
    # re-fit noise on the cleaned image (banding inflates the first estimate)
    a, b = estimate_noise_model(cleaned)
    sigma = noise_sigma_map(cleaned, a, b)
    return {"cleaned": cleaned, "sigma": sigma, "a": a, "b": b}


# ---------------------------------------------------------------------------
# 4. Classical denoiser (fallback): generalised Anscombe VST + NLM
# ---------------------------------------------------------------------------
def gat_forward(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Generalised Anscombe transform: makes Poisson-Gaussian noise ~unit variance."""
    if a < 1e-6:
        return x / np.sqrt(b)
    return (2.0 / a) * np.sqrt(np.maximum(a * x + (3.0 / 8.0) * a * a + b, 0.0))


def gat_inverse(y: np.ndarray, a: float, b: float) -> np.ndarray:
    """Algebraic inverse of the generalised Anscombe transform."""
    if a < 1e-6:
        return y * np.sqrt(b)
    return ((a * y / 2.0) ** 2 - (3.0 / 8.0) * a * a - b) / a


def classical_denoise(img: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """Full classical pipeline: stages 1-3, then NLM in the variance-stabilised
    domain where a single filtering strength suits every intensity level."""
    pre = preprocess(img)
    x, a, b = pre["cleaned"], pre["a"], pre["b"]
    y = gat_forward(np.clip(x, 0, 1), a, b)
    lo, hi = float(y.min()), float(y.max())
    scale = 255.0 / max(hi - lo, 1e-6)          # unit noise std -> `scale` grey levels
    y8 = np.clip((y - lo) * scale, 0, 255).round().astype(np.uint8)
    h = float(strength * 1.0 * scale)
    den8 = cv2.fastNlMeansDenoisingColored(y8, None, h, h, 5, 15)
    y_hat = den8.astype(np.float32) / scale + lo
    return np.clip(gat_inverse(y_hat, a, b), 0, 1).astype(np.float32)
