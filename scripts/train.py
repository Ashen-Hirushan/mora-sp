"""
train.py
-----------------------
Trains the NAFNet stage of the pipeline (GPU recommended, e.g. Colab T4).

Key design decisions
  * Inputs are produced by exactly the same classical pre-processing used at
    inference (classical.preprocess), computed once on the full 992x992
    image and cached to disk, so row/column banding and noise-model estimates
    use full-image statistics even though training uses crops.
  * Loss = -(0.04 * PSNR + 0.4 * SSIM): the gradient of the official score
    0.6*dPSNR/15 + 0.4*dSSIM with respect to the output, i.e. the network is
    trained directly on the competition metric. SSIM uses the same settings
    as evaluation/evaluate.py (7x7 uniform window, sample covariance).
  * Fixed, noise-stratified 40-image validation split (splits/val_ids.txt),
    scored with the official evaluation functions.
  * AdamW + cosine LR, AMP, EMA of weights, progressive crop size,
    automatic resume from the last checkpoint (Colab disconnects).

USAGE (Colab):
    python scripts/train.py --noisy_dir /content/data/noisy \
        --gt_dir /content/data/ground_truth --work_dir /content/drive/MyDrive/mora_run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR / "evaluation"))

from classical import preprocess  # noqa: E402
from model import PRESETS, build_model  # noqa: E402
from pipeline import SIGMA_SCALE, load_rgb, run_network  # noqa: E402

H = W = 992


# ---------------------------------------------------------------------------
# Data cache (memory-mapped, built once)
# ---------------------------------------------------------------------------
def _prepare_one(job):
    idx, noisy_path, gt_path = job
    noisy = load_rgb(Path(noisy_path))
    gt = load_rgb(Path(gt_path))
    pre = preprocess(noisy)
    return idx, noisy, gt, pre["cleaned"], pre["sigma"]


def build_cache(ids: list[str], noisy_dir: Path, gt_dir: Path, cache_dir: Path,
                workers: int) -> None:
    done_flag = cache_dir / "done.json"
    if done_flag.exists() and json.loads(done_flag.read_text())["ids"] == ids:
        print(f"Cache found in {cache_dir}")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    shapes = {"noisy": (n, H, W, 3), "gt": (n, H, W, 3), "cleaned": (n, H, W, 3), "sigma": (n, H, W)}
    # noisy/gt keep their exact 8-bit PNG values; derived channels use float16
    dtypes = {"noisy": np.uint8, "gt": np.uint8, "cleaned": np.float16, "sigma": np.float16}
    arrays = {k: np.lib.format.open_memmap(cache_dir / f"{k}.npy", mode="w+",
                                           dtype=dtypes[k], shape=shapes[k]) for k in shapes}
    jobs = [(i, str(noisy_dir / f"{s}_noise.png"), str(gt_dir / f"{s}.png")) for i, s in enumerate(ids)]
    t0 = time.time()
    with Pool(workers) as pool:
        for k, (i, noisy, gt, cleaned, sigma) in enumerate(pool.imap_unordered(_prepare_one, jobs)):
            arrays["noisy"][i] = np.round(noisy * 255).astype(np.uint8)
            arrays["gt"][i] = np.round(gt * 255).astype(np.uint8)
            arrays["cleaned"][i], arrays["sigma"][i] = cleaned, sigma
            if (k + 1) % 20 == 0 or k + 1 == n:
                print(f"  cached {k + 1}/{n}  ({time.time() - t0:.0f}s)")
    for a in arrays.values():
        a.flush()
    done_flag.write_text(json.dumps({"ids": ids}))


class CropStream(IterableDataset):
    """Infinite stream of random crops from the memory-mapped cache."""

    def __init__(self, cache_dir: Path, indices: list[int], crop: int, seed: int):
        self.cache_dir, self.indices, self.crop, self.seed = cache_dir, indices, crop, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) + int(time.time()))
        arr = {k: np.load(self.cache_dir / f"{k}.npy", mmap_mode="r")
               for k in ("noisy", "gt", "cleaned", "sigma")}
        c = self.crop
        while True:
            i = self.indices[rng.integers(len(self.indices))]
            y, x = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
            inp = np.concatenate([arr["noisy"][i, y:y + c, x:x + c] / np.float32(255),
                                  arr["cleaned"][i, y:y + c, x:x + c],
                                  arr["sigma"][i, y:y + c, x:x + c, None] * SIGMA_SCALE], axis=2)
            gt = arr["gt"][i, y:y + c, x:x + c] / np.float32(255)
            yield (torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1), dtype=np.float32)),
                   torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1), dtype=np.float32)))


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def load_or_make_split(ids: list[str], noisy_dir: Path, gt_dir: Path, n_val: int) -> list[str]:
    """Noise-stratified split: sort images by noisy-input PSNR and take evenly
    spaced ones, so validation covers every noise severity."""
    split_file = SCRIPT_DIR / "splits" / "val_ids.txt"
    if split_file.exists():
        return split_file.read_text().split()
    psnrs = []
    for s in ids:
        n, g = load_rgb(noisy_dir / f"{s}_noise.png"), load_rgb(gt_dir / f"{s}.png")
        psnrs.append(-10 * np.log10(np.mean((n - g) ** 2)))
    order = [ids[i] for i in np.argsort(psnrs)]
    step = len(order) / n_val
    val = sorted(order[int(step * (k + 0.5))] for k in range(n_val))
    split_file.parent.mkdir(exist_ok=True)
    split_file.write_text("\n".join(val) + "\n")
    return val


# ---------------------------------------------------------------------------
# Loss = negative competition score
# ---------------------------------------------------------------------------
def ssim_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample SSIM matching skimage(win_size=7, gaussian_weights=False,
    use_sample_covariance=True, data_range=1)."""
    c1, c2, k, npix = 0.01 ** 2, 0.03 ** 2, 7, 49
    cov_norm = npix / (npix - 1)
    ux, uy = F.avg_pool2d(x, k, 1), F.avg_pool2d(y, k, 1)
    uxx, uyy, uxy = F.avg_pool2d(x * x, k, 1), F.avg_pool2d(y * y, k, 1), F.avg_pool2d(x * y, k, 1)
    vx, vy, vxy = cov_norm * (uxx - ux * ux), cov_norm * (uyy - uy * uy), cov_norm * (uxy - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return s.mean(dim=(1, 2, 3))


def score_loss(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    pred = pred.float()
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse + 1e-10)
    ssim = ssim_torch(pred, gt)
    loss = -(0.04 * psnr + 0.4 * ssim).mean()
    return loss, psnr.mean().item(), ssim.mean().item()


def augment(inp: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Random dihedral transform (same for the whole batch, applied on GPU)."""
    k = int(torch.randint(4, (1,)))
    if torch.rand(1) < 0.5:
        inp, gt = inp.flip(3), gt.flip(3)
    return torch.rot90(inp, k, (2, 3)), torch.rot90(gt, k, (2, 3))


# ---------------------------------------------------------------------------
# Validation with the official metric
# ---------------------------------------------------------------------------
_NOISY_METRICS: dict[int, tuple[float, float]] = {}   # noisy-input PSNR/SSIM never change


def validate(model, cache_dir: Path, val_idx: list[int], device, tta: int = 1) -> dict:
    from evaluate import official_composite, ssim_value
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr

    arr = {k: np.load(cache_dir / f"{k}.npy", mmap_mode="r") for k in ("noisy", "gt", "cleaned", "sigma")}
    model.eval()
    rows = []
    for i in val_idx:
        noisy8 = arr["noisy"][i].astype(np.float32) / 255
        gt8 = arr["gt"][i].astype(np.float32) / 255
        inp = np.concatenate([noisy8, arr["cleaned"][i].astype(np.float32),
                              arr["sigma"][i].astype(np.float32)[..., None] * SIGMA_SCALE], axis=2)
        x = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1)))[None].to(device)
        pred = run_network(model, x, tta)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        pred = np.round(pred * 255) / 255      # 8-bit, exactly as saved to PNG
        if i not in _NOISY_METRICS:
            _NOISY_METRICS[i] = (sk_psnr(gt8, noisy8, data_range=1.0), ssim_value(gt8, noisy8))
        noisy_psnr, noisy_ssim = _NOISY_METRICS[i]
        dp = sk_psnr(gt8, pred, data_range=1.0) - noisy_psnr
        ds = ssim_value(gt8, pred) - noisy_ssim
        rows.append((dp, ds, official_composite(dp, ds)))
    model.train()
    r = np.array(rows)
    return {"dpsnr": float(r[:, 0].mean()), "dssim": float(r[:, 1].mean()), "score": float(r[:, 2].mean())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--noisy_dir", type=Path, default=REPO_DIR / "competition_data/public/noisy")
    ap.add_argument("--gt_dir", type=Path, default=REPO_DIR / "competition_data/public/ground_truth")
    ap.add_argument("--work_dir", type=Path, required=True, help="checkpoints + logs (use Google Drive on Colab)")
    ap.add_argument("--cache_dir", type=Path, default=None, help="pre-processed cache (default: <work_dir>/cache; "
                    "on Colab prefer local disk, e.g. /content/cache)")
    ap.add_argument("--preset", choices=list(PRESETS), default="small")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--final_crop", type=int, default=384, help="crop size for the last --final_frac of training")
    ap.add_argument("--final_frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val_every", type=int, default=2500)
    ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--init", type=Path, default=None,
                    help="initialise from exported weights (e.g. fine-tuning on all images)")
    ap.add_argument("--train_on_all", action="store_true", help="also train on the validation images (final model)")
    ap.add_argument("--max_hours", type=float, default=0, help="save and stop after this many hours (0 = off)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.work_dir / "cache"

    ids = sorted(p.stem for p in args.gt_dir.glob("*.png"))
    if args.limit:
        ids = ids[: args.limit]
    val_ids = [s for s in load_or_make_split(ids, args.noisy_dir, args.gt_dir, args.n_val) if s in ids]
    if not val_ids:                               # smoke tests with --limit
        val_ids = ids[-2:]
    build_cache(ids, args.noisy_dir, args.gt_dir, cache_dir, args.workers)
    val_idx = [ids.index(s) for s in val_ids]
    train_idx = list(range(len(ids))) if args.train_on_all else [i for i in range(len(ids)) if i not in val_idx]
    print(f"device={device} train={len(train_idx)} val={len(val_idx)} preset={args.preset}")

    model = build_model(args.preset).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-4)
    lr_at = lambda it: args.lr * min(1.0, (it + 1) / args.warmup) * \
        (1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * it / args.iters)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_it, best = 0, -1.0
    last_ckpt = args.work_dir / "last.pt"
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_it, best = ck["iter"], ck["best"]
        print(f"Resumed from iteration {start_it} (best val score {best:.4f})")
    elif args.init:
        sd = torch.load(args.init, map_location=device, weights_only=True)["state_dict"]
        model.load_state_dict(sd)
        ema.load_state_dict(sd)
        print(f"Initialised from {args.init}")

    def make_loader(crop, batch):
        return iter(DataLoader(CropStream(cache_dir, train_idx, crop, args.seed + start_it),
                               batch_size=batch, num_workers=min(args.workers, 4),
                               pin_memory=device.type == "cuda", persistent_workers=False))

    final_start = int(args.iters * (1 - args.final_frac))
    stage_final = start_it >= final_start
    crop, batch = (args.final_crop, max(1, args.batch * args.crop ** 2 // args.final_crop ** 2)) \
        if stage_final else (args.crop, args.batch)
    loader = make_loader(crop, batch)

    def save(it):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "iter": it, "best": best, "args": vars(args)},
                   args.work_dir / "last.tmp")
        os.replace(args.work_dir / "last.tmp", last_ckpt)

    def export(path):
        sd = {k: v.detach().float().cpu() for k, v in ema.state_dict().items()}
        torch.save({"config": ema.config, "state_dict": sd}, path)

    log = open(args.work_dir / "log.txt", "a")
    t0, run_start = time.time(), time.time()
    model.train()
    for it in range(start_it, args.iters):
        if not stage_final and it >= final_start:
            stage_final = True
            crop, batch = args.final_crop, max(1, args.batch * args.crop ** 2 // args.final_crop ** 2)
            loader = make_loader(crop, batch)
            print(f"Switching to crop {crop}, batch {batch}")

        inp, gt = next(loader)
        inp, gt = augment(inp.to(device, non_blocking=True), gt.to(device, non_blocking=True))
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(inp)
        loss, psnr, ssim = score_loss(pred, gt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            decay = min(args.ema, (it + 1) / (it + 10))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        if (it + 1) % 100 == 0:
            speed = 100 / (time.time() - t0)
            eta = (args.iters - it - 1) / speed / 3600
            msg = (f"it {it + 1}/{args.iters} loss {loss.item():.4f} crop-PSNR {psnr:.2f} "
                   f"SSIM {ssim:.4f} lr {lr_at(it):.2e} {speed:.2f} it/s ETA {eta:.1f}h")
            print(msg); log.write(msg + "\n"); log.flush()
            t0 = time.time()

        last = it + 1 == args.iters
        if (it + 1) % args.val_every == 0 or last:
            v = validate(ema, cache_dir, val_idx, device)
            msg = (f"[val] it {it + 1}: dPSNR {v['dpsnr']:+.3f} dB  dSSIM {v['dssim']:+.4f}  "
                   f"SCORE {v['score']:.4f}")
            print(msg); log.write(msg + "\n"); log.flush()
            if v["score"] > best or args.train_on_all:
                best = v["score"]
                export(args.work_dir / f"nafnet_{args.preset}.pt")
                print(f"  exported weights -> {args.work_dir / f'nafnet_{args.preset}.pt'}")
            save(it + 1)
            t0 = time.time()

        if args.max_hours and time.time() - run_start > args.max_hours * 3600:
            save(it + 1)
            print(f"Time budget reached at iteration {it + 1}; checkpoint saved. Re-run to resume.")
            return

    print(f"Training finished. Best validation score: {best:.4f}")


if __name__ == "__main__":
    main()
