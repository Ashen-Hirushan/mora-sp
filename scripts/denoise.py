"""
denoise.py
-----------------------
Mora SP Cup 2026 -- low-light image denoising (submission entry point).

Hybrid pipeline:
    1. blind Poisson-Gaussian noise-model estimation   (classical)
    2. noise-adaptive defect-pixel correction          (classical)
    3. row / column banding removal                    (classical)
    4. lightweight NAFNet conditioned on the noisy image, the cleaned
       image and the estimated noise map               (deep learning)

If no trained weights are found, a fully classical fallback is used
(generalised Anscombe transform + Non-Local Means).

Reads every `<id>_noise.png` in --noise_dir and writes `<id>.png` to
--denoised_dir. Runs on CPU by default; uses CUDA automatically if present.

USAGE:
    python scripts/denoise.py --noise_dir competition_data/submissions/noisy \
                              --denoised_dir competition_data/submissions/denoised
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from classical import classical_denoise  # noqa: E402
from pipeline import denoise_image, load_network, load_rgb, save_rgb  # noqa: E402

VALID_EXT = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights" / "nafnet_small.pt"


def output_name(path: Path) -> str:
    """461_noise.png -> 461.png"""
    stem = path.stem
    if stem.lower().endswith(NOISE_SUFFIX):
        stem = stem[: -len(NOISE_SUFFIX)]
    return f"{stem}.png"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--noise_dir", "--input_dir", dest="noise_dir", required=True, type=Path,
                    help="folder containing <id>_noise.png images")
    ap.add_argument("--denoised_dir", "--output_dir", dest="denoised_dir", required=True, type=Path,
                    help="folder where <id>.png outputs are written")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="trained network checkpoint (default: scripts/weights/nafnet_small.pt)")
    ap.add_argument("--method", choices=["auto", "network", "classical"], default="auto",
                    help="auto = network if weights exist, otherwise classical fallback")
    ap.add_argument("--tta", type=int, choices=[1, 2, 4, 8], default=1,
                    help="geometric self-ensemble size (higher = slightly better, slower)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--threads", type=int, default=os.cpu_count(),
                    help="CPU threads used by PyTorch")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available -- falling back to CPU.")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    method = args.method
    if method == "auto":
        method = "network" if args.weights.exists() else "classical"
        if method == "classical":
            print(f"WARNING: weights not found at {args.weights}; using classical fallback.")
    if method == "network" and not args.weights.exists():
        raise SystemExit(f"ERROR: weights not found: {args.weights}")

    paths = sorted(p for p in args.noise_dir.glob("*") if p.suffix.lower() in VALID_EXT)
    if not paths:
        raise SystemExit(f"ERROR: no images found in {args.noise_dir}")
    args.denoised_dir.mkdir(parents=True, exist_ok=True)

    model = load_network(args.weights, device) if method == "network" else None
    print(f"Method: {method} | device: {device} | TTA: {args.tta if model else '-'} | "
          f"images: {len(paths)}")

    total = 0.0
    for path in paths:
        noisy = load_rgb(path)
        start = time.perf_counter()
        if model is not None:
            result = denoise_image(noisy, model, device, args.tta)
        else:
            result = classical_denoise(noisy)
        elapsed = time.perf_counter() - start
        total += elapsed
        out_path = args.denoised_dir / output_name(path)
        save_rgb(result, out_path)
        print(f"  {path.name} -> {out_path.name}  ({elapsed:.2f}s)")

    print(f"Processed {len(paths)} images in {total:.1f}s "
          f"({total / len(paths):.2f}s/image). Results in {args.denoised_dir}")


if __name__ == "__main__":
    main()
