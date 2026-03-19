"""
source ~/.bashrc
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import kaggle

DATA_ROOT = Path("data")

DIRS = {
    "rsna":          DATA_ROOT / "rsna_bone_age",
    "nih":           DATA_ROOT / "nih_chest_xray14",
    "chexpert":      DATA_ROOT / "chexpert",
    "fracatlas":     DATA_ROOT / "fracatlas",
    "fracture_msk":  DATA_ROOT / "fracture_msk",
    "stanford":      DATA_ROOT / "stanford_bone_age",
    "mura":          DATA_ROOT / "MURA-v1.1",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def header(msg):
    print(f"\n{'─'*64}")
    print(f"  {msg}")
    print(f"{'─'*64}")

def count_images(directory):
    if not Path(directory).exists():
        return 0
    return sum(1 for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_EXTS)

def verify_dataset(name, directory, expected_min):
    n = count_images(directory)
    ok = n >= expected_min
    print(f"  {'✓' if ok else '✗'}  {name:<35} {n:>8,} images  (expected ≥ {expected_min:,})")
    return ok

def kaggle_download(dataset_slug, dest: Path, expected_min: int, size_hint: str):
    """Download a Kaggle dataset and flatten images into dest/images/."""
    if count_images(dest) >= expected_min:
        print(f"  ✓  Already downloaded ({count_images(dest):,} images)")
        return True

    dest.mkdir(parents=True, exist_ok=True)
    tmp = DATA_ROOT / f"_tmp_{dest.name}"
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {dataset_slug} (~{size_hint}) ...")
    ret = os.system(f"kaggle datasets download -d {dataset_slug} -p {tmp} --unzip")
    if ret != 0:
        print(f"  ✗  kaggle download failed (exit {ret})")
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    images_out = dest / "images"
    images_out.mkdir(exist_ok=True)
    moved = 0
    for img in tmp.rglob("*"):
        if img.suffix.lower() in IMAGE_EXTS:
            # preserve relative path as flat name to avoid collisions
            # e.g. patient1/study1/frontal.jpg → patient1_study1_frontal.jpg
            rel = img.relative_to(tmp)
            safe_name = "_".join(rel.parts)
            target = images_out / safe_name
            if not target.exists():
                shutil.copy2(img, target)
            moved += 1

    labels_out = dest / "labels"
    labels_out.mkdir(exist_ok=True)
    for csv in tmp.rglob("*.csv"):
        shutil.copy2(csv, labels_out / csv.name)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  ✓  {moved:,} images → {images_out}")
    return moved > 0


# ── Individual dataset downloaders ────────────────────────────────────────────

def download_rsna():
    header("1/4  RSNA Pediatric Bone Age (~14K hand/wrist X-rays, ~9.6 GB)")
    return kaggle_download("kmader/rsna-bone-age", DIRS["rsna"], 12_000, "9.6 GB")

def download_nih():
    header("2/5  NIH ChestX-ray14 (~112K chest X-rays, ~42 GB)")
    print("  Note: requires accepting terms at https://www.kaggle.com/datasets/nih-chest-xrays/data")
    return kaggle_download("nih-chest-xrays/data", DIRS["nih"], 100_000, "42 GB")

def download_chexpert():
    header("5/5  CheXpert (~224K chest X-rays, ~11 GB)")
    return kaggle_download("ashery/chexpert", DIRS["chexpert"], 200_000, "~11 GB")

def download_fracatlas():
    header("3/4  FracAtlas (~4K MSK X-rays — hands, legs, hips)")
    return kaggle_download("tommyngx/fracatlas", DIRS["fracatlas"], 3_500, "~800 MB")

def download_fracture_msk():
    header("4/4  Bone Fracture Multi-Region (~4K MSK — wrist, hand, elbow, shoulder)")
    return kaggle_download(
        "bmadushanirodrigo/fracture-multi-region-x-ray-data",
        DIRS["fracture_msk"], 3_000, "~600 MB"
    )


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary():
    header("Dataset Summary")
    verify_dataset("MURA v1.1 (MSK)",           DIRS["mura"],        36_000)
    verify_dataset("RSNA Bone Age (MSK)",        DIRS["rsna"],        12_000)
    verify_dataset("FracAtlas (MSK)",            DIRS["fracatlas"],    3_500)
    verify_dataset("Fracture Multi-Region (MSK)",DIRS["fracture_msk"], 3_000)
    verify_dataset("NIH ChestX-ray14 (CXR)",     DIRS["nih"],        100_000)
    verify_dataset("Stanford Bone Age (MSK)",    DIRS["stanford"],    5_000)

    msk = (count_images(DIRS["mura"]) + count_images(DIRS["rsna"]) +
           count_images(DIRS["fracatlas"]) + count_images(DIRS["fracture_msk"]) +
           count_images(DIRS["stanford"]))
    cxr = count_images(DIRS["nih"])

    print(f"""
  MSK images (in-domain):  {msk:>8,}
  CXR images (off-domain): {cxr:>8,}
  ─────────────────────────────────────
  Total pretraining corpus:{msk+cxr:>8,}
""")

    disk_gb = sum(
        sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
        for d in DIRS.values() if Path(d).exists()
    ) / 1e9
    print(f"  Disk used: {disk_gb:.1f} GB\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nih", action="store_true", help="Skip NIH CXR14 (saves ~42 GB)")
    parser.add_argument("--msk-only", action="store_true", help="MSK datasets only, skip NIH")
    parser.add_argument("--verify",   action="store_true", help="Verify only, no downloads")
    args = parser.parse_args()

    DATA_ROOT.mkdir(exist_ok=True)

    if args.verify:
        print_summary()
        return


    download_rsna()
    download_fracatlas()
    download_fracture_msk()

    if not args.skip_nih and not args.msk_only:
        download_nih()
        download_chexpert()
    else:
        print("\n  Skipping NIH ChestX-ray14 and CheXpert.")

    print_summary()
    print(f"""
{'─'*64}
  Next step:
      python pretrain_mae.py             # use all available data
      python pretrain_mae.py --msk-only  # MSK only (more domain-specific)
{'─'*64}
""")


if __name__ == "__main__":
    main()