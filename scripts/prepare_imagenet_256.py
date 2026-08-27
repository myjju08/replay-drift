#!/usr/bin/env python3
"""Prepare ImageNet-2012 for 256x256 latent-space training.

This script takes the standard official ImageNet 2012 archives:
  - ILSVRC2012_devkit_t12.tar.gz
  - ILSVRC2012_img_train.tar
  - ILSVRC2012_img_val.tar

It then:
  1. extracts them into torchvision-compatible `train/` and `val/` folders
  2. builds the SD-VAE latent cache used by generator training

It also supports the Kaggle `imagenet-object-localization-challenge` layout,
which unpacks to:
  - ILSVRC/Data/CLS-LOC/train
  - ILSVRC/Data/CLS-LOC/val
  - LOC_val_solution.csv

Default output paths are repo-local so the prepared data can be used without
requiring `/data/...` on the host.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import torch
from torchvision.datasets.imagenet import parse_devkit_archive, parse_train_archive, parse_val_archive

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "imagenet"
DEFAULT_IMAGENET_ROOT = DEFAULT_DATA_ROOT / "ILSVRC2012"
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "latent_cache_sd_vae_mse"


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _has_any_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _clear_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _is_kaggle_layout(raw_root: Path) -> bool:
    return (
        (raw_root / "ILSVRC" / "Data" / "CLS-LOC" / "train").is_dir()
        and (raw_root / "ILSVRC" / "Data" / "CLS-LOC" / "val").is_dir()
        and (raw_root / "LOC_val_solution.csv").is_file()
    )


def _symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src)
    except OSError:
        # Fallback for filesystems where symlinks are restricted.
        shutil.copy2(src, dst)


def _prepare_imagenet_tree(
    *,
    raw_root: Path,
    imagenet_root: Path,
    force_extract: bool,
) -> None:
    if _is_kaggle_layout(raw_root):
        _prepare_kaggle_tree(
            raw_root=raw_root,
            imagenet_root=imagenet_root,
            force_extract=force_extract,
        )
        return

    devkit = _require_file(raw_root / "ILSVRC2012_devkit_t12.tar.gz", "ImageNet devkit archive")
    train_tar = _require_file(raw_root / "ILSVRC2012_img_train.tar", "ImageNet train archive")
    val_tar = _require_file(raw_root / "ILSVRC2012_img_val.tar", "ImageNet val archive")

    imagenet_root.mkdir(parents=True, exist_ok=True)
    train_dir = imagenet_root / "train"
    val_dir = imagenet_root / "val"
    meta_file = imagenet_root / "meta.bin"

    if force_extract:
        if meta_file.exists():
            meta_file.unlink()

    if force_extract or not meta_file.exists():
        print(f"[prep] parsing devkit -> {meta_file}")
        parse_devkit_archive(imagenet_root, file=str(devkit))
    else:
        print(f"[skip] devkit already parsed: {meta_file}")

    if force_extract or not _has_any_files(train_dir):
        print(f"[prep] extracting train archive -> {train_dir}")
        parse_train_archive(imagenet_root, file=str(train_tar), folder="train")
    else:
        print(f"[skip] train split already exists: {train_dir}")

    if force_extract or not _has_any_files(val_dir):
        print(f"[prep] extracting val archive -> {val_dir}")
        parse_val_archive(imagenet_root, file=str(val_tar), folder="val")
    else:
        print(f"[skip] val split already exists: {val_dir}")


def _prepare_kaggle_tree(
    *,
    raw_root: Path,
    imagenet_root: Path,
    force_extract: bool,
) -> None:
    source_train = _require_dir(
        raw_root / "ILSVRC" / "Data" / "CLS-LOC" / "train",
        "Kaggle ImageNet train directory",
    )
    source_val = _require_dir(
        raw_root / "ILSVRC" / "Data" / "CLS-LOC" / "val",
        "Kaggle ImageNet val directory",
    )
    val_solution = _require_file(raw_root / "LOC_val_solution.csv", "Kaggle val solution CSV")

    imagenet_root.mkdir(parents=True, exist_ok=True)
    train_dir = imagenet_root / "train"
    val_dir = imagenet_root / "val"

    if force_extract:
        if train_dir.exists() or train_dir.is_symlink():
            _clear_path(train_dir)
        if val_dir.exists() or val_dir.is_symlink():
            _clear_path(val_dir)

    if not train_dir.exists():
        print(f"[prep] linking Kaggle train split -> {train_dir}")
        train_dir.symlink_to(source_train, target_is_directory=True)
    else:
        print(f"[skip] train split already exists: {train_dir}")

    if force_extract or not _has_any_files(val_dir):
        if val_dir.exists() or val_dir.is_symlink():
            _clear_path(val_dir)
        val_dir.mkdir(parents=True, exist_ok=True)
        print(f"[prep] organizing Kaggle val split -> {val_dir}")
        with val_solution.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row["ImageId"]
                pred = (row["PredictionString"] or "").strip()
                if not pred:
                    continue
                wnid = pred.split()[0]
                src = source_val / f"{image_id}.JPEG"
                if not src.exists():
                    raise FileNotFoundError(f"Missing val image referenced by CSV: {src}")
                dst_dir = val_dir / wnid
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                if not dst.exists():
                    _symlink_or_copy(src, dst)
    else:
        print(f"[skip] val split already exists: {val_dir}")


def _build_cache(
    *,
    imagenet_root: Path,
    cache_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    resolution: int,
) -> None:
    from vae_imagenet import build_latent_cache

    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"[cache] building latent cache at {cache_root}")
    print(f"[cache] source={imagenet_root} resolution={resolution} batch_size={batch_size} device={device}")
    build_latent_cache(
        data_path=str(imagenet_root),
        target_path=str(cache_root),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        resolution=resolution,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ImageNet 256x256 latent cache for DualDrift")
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing either the official ImageNet archives or the Kaggle ILSVRC layout.",
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=DEFAULT_IMAGENET_ROOT,
        help="Output directory for extracted ImageNet train/ and val/ trees.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Output directory for the SD-VAE latent cache.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Center-crop resolution for cache creation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for VAE latent encoding.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers for cache creation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Encoding device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only prepare train/ and val/ folders; skip latent cache creation.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Skip extraction and only build the latent cache from an existing ImageNet tree.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run archive extraction even if train/val/meta already exist.",
    )
    args = parser.parse_args()

    if args.extract_only and args.cache_only:
        raise ValueError("--extract-only and --cache-only cannot be used together.")

    raw_root = args.raw_root.resolve()
    imagenet_root = args.imagenet_root.resolve()
    cache_root = args.cache_root.resolve()
    device = _resolve_device(args.device)

    if not args.cache_only:
        _prepare_imagenet_tree(
            raw_root=raw_root,
            imagenet_root=imagenet_root,
            force_extract=args.force_extract,
        )

    if not args.extract_only:
        _build_cache(
            imagenet_root=imagenet_root,
            cache_root=cache_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            resolution=args.resolution,
        )

    print("[OK] ImageNet 256 preparation complete.")
    print(f"[OK] imagenet_root={imagenet_root}")
    print(f"[OK] cache_root={cache_root}")


if __name__ == "__main__":
    main()
