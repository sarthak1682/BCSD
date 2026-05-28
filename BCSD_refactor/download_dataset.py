"""
Download bench dataset files from Google Drive.

Usage:
    python download_dataset.py --output_dir /path/to/nvemb
    python download_dataset.py --output_dir /home/ra72yeq/projects/NovaXLLM2Vec/nvemb

Files downloaded:
    output_benchset_rebalanced_train_nova.jsonl
    output_benchset_rebalanced_test_nova.jsonl

Requires: pip install gdown
"""

import argparse
import os
import subprocess
import sys


# Google Drive folder ID (from the shared link)
DRIVE_FOLDER_ID = "1uG9nQA9Q2U-m3CcmUjxU8FpQ7cE1UUWS"

# Individual file names expected in the folder
FILES = [
    "output_benchset_rebalanced_train_nova.jsonl",
    "output_benchset_rebalanced_test_nova.jsonl",
]


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])


def download_folder(output_dir: str, force: bool = False) -> None:
    import gdown

    os.makedirs(output_dir, exist_ok=True)

    # Check which files are already present
    missing = []
    for fname in FILES:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath) and not force:
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print(f"  [skip] {fname} already exists ({size_mb:.1f} MB)")
        else:
            missing.append(fname)

    if not missing:
        print("All files already present. Use --force to re-download.")
        return

    print(f"\nDownloading {len(missing)} file(s) from Google Drive folder {DRIVE_FOLDER_ID} ...")
    print(f"Output dir: {output_dir}\n")

    # Download entire folder — gdown will create a subfolder by default,
    # so we download to a temp location and move files up.
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmp:
        url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
        gdown.download_folder(url, output=tmp, quiet=False, use_cookies=False)

        # gdown creates a subfolder named after the Drive folder ("nvemb")
        # Find whatever subfolder was created
        subdirs = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
        src_dir = os.path.join(tmp, subdirs[0]) if subdirs else tmp

        for fname in FILES:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src):
                shutil.move(src, dst)
                size_mb = os.path.getsize(dst) / (1024 ** 2)
                print(f"  [ok] {fname} → {dst} ({size_mb:.1f} MB)")
            else:
                print(f"  [warn] {fname} not found in downloaded folder — check Drive permissions")


def verify(output_dir: str) -> bool:
    print("\nVerification:")
    all_ok = True
    for fname in FILES:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            # Sanity-check: count lines
            with open(fpath) as f:
                n_lines = sum(1 for _ in f)
            print(f"  ✓ {fname}: {n_lines:,} samples ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {fname}: MISSING at {fpath}")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download bench dataset from Google Drive")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../")),
            "nvemb"
        ),
        help="Directory to save the .jsonl files (default: repo_root/nvemb/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    ensure_gdown()
    download_folder(args.output_dir, force=args.force)
    ok = verify(args.output_dir)

    if ok:
        print("\nDataset ready.")
        print(f"Train: {os.path.join(args.output_dir, FILES[0])}")
        print(f"Test:  {os.path.join(args.output_dir, FILES[1])}")
    else:
        print("\nSome files are missing. Check Drive permissions or download manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
