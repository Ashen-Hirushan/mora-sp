# Scripts — Hybrid Classical + NAFNet Low-Light Denoiser

> **Important — Submission Freeze**
>
> After the preliminary-round submission deadline, no new commits may be
> made to the submitted private GitHub repository.
>
> The Git commit SHA stated in this README and recorded by the organizers
> will be treated as the official submitted code version.
>
> If external model weights/checkpoints are used, their Google Drive link,
> expected local path, and SHA-256 checksum must also be stated here.
> The submitted model/checkpoint must not be modified or replaced after
> the deadline.

## Method overview

```
noisy RGB ──► 1. Poisson-Gaussian noise-model fit  σ²(I) = a·I + b      (classical)
          ──► 2. noise-adaptive defect-pixel correction (k·σ, isolated peaks)
          ──► 3. row / column banding removal (robust line medians)
          ──► 4. NAFNet(noisy ‖ cleaned ‖ σ-map) → residual added to cleaned  (deep learning)
          ──► 8-bit RGB PNG
```

| File | Purpose |
|---|---|
| `denoise.py` | **Submission entry point** (CPU by default, CUDA if available) |
| `classical.py` | Noise estimation, defect correction, banding removal, classical GAT + NLM fallback |
| `model.py` | Lightweight NAFNet (`tiny` / `small` / `base` presets) |
| `pipeline.py` | Shared inference path used by `denoise.py` and `train.py` |
| `train.py` | GPU training (score-aligned loss, AMP, EMA, resume) |
| `colab_train.ipynb` | Colab T4 notebook that runs `train.py` |
| `package_submission.py` | Checks the 20 outputs, builds `TeamName.zip`, prints SHA values |
| `splits/val_ids.txt` | 40 held-out validation images, stratified by noise level |
| `weights/nafnet_small.pt` | Trained network weights |

If the weights file is missing, `denoise.py` prints a warning and uses the
fully classical fallback: steps 1–3, then a generalised Anscombe transform and
Non-Local Means.

## How to run

### Requirements

* Python 3.10+ (tested with Python 3.14, PyTorch 2.14 CPU)
* Packages: see `requirements.txt`
* Hardware: any x86-64 CPU. No GPU is needed for inference.

```bash
pip install -r scripts/requirements.txt
```

### Usage

From the repository root:

```bash
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
```

| Option | Default | Description |
|---|---|---|
| `--noise_dir` (alias `--input_dir`) | required | Folder containing `<id>_noise.png` |
| `--denoised_dir` (alias `--output_dir`) | required | Output folder (created if missing) |
| `--weights` | `scripts/weights/nafnet_small.pt` | Network checkpoint |
| `--method` | `auto` | `auto`, `network` or `classical` |
| `--tta` | `1` | Geometric self-ensemble size: 1, 2, 4 or 8 |
| `--device` | `auto` | `auto`, `cpu` or `cuda` |
| `--threads` | all cores | CPU threads for PyTorch |

The script runs fully offline.

### Output

Each `<id>_noise.png` becomes `<id>.png` (for example `461_noise.png` → `461.png`):
an 8-bit RGB PNG of 992×992 pixels, the same size as the input.

## Reproducing the results

```bash
# score on the held-out images: copy the ids listed in splits/val_ids.txt into
# val/noisy and val/gt, then (add --method classical for the classical pipeline)
python scripts/denoise.py --noise_dir val/noisy --denoised_dir val/out
python evaluation/evaluate.py --noisy_dir val/noisy --pred_dir val/out --gt_dir val/gt

# training (GPU): see colab_train.ipynb, or
python scripts/train.py --work_dir runs/small --preset small --iters 40000

# check the submission and build the ZIP
python scripts/package_submission.py --team TeamName
```

Validation strategy: 40 of the 460 public pairs are held out
(`splits/val_ids.txt`). They were chosen by sorting all images by noisy-input
PSNR and taking evenly spaced ones, so every noise level is represented. These
images are never used to train the reported model.

## Official Submission Information

Git Commit SHA:
a3d67fdd432b630bec895383dd1a8709d517e72b

Model Checkpoint:
`nafnet_small.pt`

Model Drive Link:
N/A committed to the repository>

Expected Model Path:
`scripts/weights/nafnet_small.pt`

Model SHA-256:
bb1245bcb2eb140cfa193a85e000b46e85c0776d7ff531bc202e4f15bac889c9 
