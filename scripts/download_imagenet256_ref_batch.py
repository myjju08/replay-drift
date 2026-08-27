"""Download the official ImageNet-256 ADM/OpenAI reference batch.

Default artifact:
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/VIRTUAL_imagenet256_labeled.npz

This .npz is the labeled 256x256 ImageNet reference batch used by the OpenAI
ADM/guided-diffusion evaluation protocol.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/"
    "jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
)


def _inspect_npz(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        print(f"[inspect] members={names}")
        for info in zf.infolist():
            size_mib = info.file_size / (2 ** 20)
            print(f"[inspect] {info.filename}: {size_mib:.1f} MiB")


def download(url: str, out_path: Path, force: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[skip] already exists: {out_path}")
        _inspect_npz(out_path)
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {url}")
    print(f"[download] -> {out_path}")
    with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.replace(out_path)
    size_mib = out_path.stat().st_size / (2 ** 20)
    print(f"[ok] downloaded {size_mib:.1f} MiB")
    _inspect_npz(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official ImageNet-256 reference batch.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/VIRTUAL_imagenet256_labeled.npz"),
        help="Output path for the downloaded .npz",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Override the download URL")
    parser.add_argument("--force", action="store_true", help="Redownload even if the output already exists")
    args = parser.parse_args()

    try:
        download(args.url, args.out, force=bool(args.force))
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
