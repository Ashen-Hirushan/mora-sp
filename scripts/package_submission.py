"""
package_submission.py
-----------------------
Checks the preliminary-round outputs against the submission rules and builds
TeamName.zip. Also prints the Git commit SHA and the SHA-256 of the model
weights for scripts/README.md.

Checks: exactly 461.png ... 480.png, nothing else, PNG format, RGB,
992x992, 8-bit.

USAGE:
    python scripts/package_submission.py --team TeamName
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
EXPECTED = [f"{i}.png" for i in range(461, 481)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True, help="team name used for the ZIP file name")
    ap.add_argument("--denoised_dir", type=Path, default=REPO_DIR / "competition_data/submissions/denoised")
    ap.add_argument("--weights", type=Path, default=SCRIPT_DIR / "weights" / "nafnet_small.pt")
    ap.add_argument("--out_dir", type=Path, default=REPO_DIR.parent,
                    help="where TeamName.zip is written (default: outside the repository)")
    args = ap.parse_args()

    present = sorted(p.name for p in args.denoised_dir.iterdir() if p.is_file())
    errors = []
    missing = [n for n in EXPECTED if n not in present]
    extra = [n for n in present if n not in EXPECTED]
    if missing:
        errors.append(f"missing files: {missing}")
    if extra:
        errors.append(f"unexpected files: {extra}")
    for name in EXPECTED:
        path = args.denoised_dir / name
        if not path.exists():
            continue
        with Image.open(path) as img:
            if img.format != "PNG":
                errors.append(f"{name}: format {img.format}, expected PNG")
            if img.mode != "RGB":
                errors.append(f"{name}: mode {img.mode}, expected RGB")
            if img.size != (992, 992):
                errors.append(f"{name}: size {img.size}, expected (992, 992)")
    if errors:
        raise SystemExit("Submission check FAILED:\n  " + "\n  ".join(errors))

    zip_path = args.out_dir / f"{args.team}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in EXPECTED:
            z.write(args.denoised_dir / name, name)       # flat: no folders inside the ZIP
    print(f"All checks passed. Wrote {zip_path} ({len(EXPECTED)} images)")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "scripts"], cwd=REPO_DIR, text=True).strip()
        print(f"Git commit SHA: {commit}" + ("   (WARNING: scripts/ has uncommitted changes)" if dirty else ""))
    except (OSError, subprocess.CalledProcessError):
        print("Git commit SHA: unavailable")
    if args.weights.exists():
        print(f"Model SHA-256 ({args.weights.name}): {sha256(args.weights)}")


if __name__ == "__main__":
    main()
