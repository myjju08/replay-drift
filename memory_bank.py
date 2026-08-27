"""ArrayMemoryBank — class-wise ring buffer for real/fake samples.

PyTorch port of the official JAX ArrayMemoryBank.
Stores samples as numpy arrays (CPU); returns torch tensors on demand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


class ArrayMemoryBank:
    """Per-class ring buffer that stores image/latent samples.

    Used during generator training to maintain a pool of real images
    (positive bank) and unconditioned images (negative bank).

    Args:
        num_classes: Number of distinct labels (1000 for ImageNet).
        max_size:    Maximum stored samples per class.
        dtype:       NumPy dtype for storage (default float32).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        max_size: int = 64,
        dtype=np.float32,
    ):
        self.num_classes   = int(num_classes)
        self.max_size      = int(max_size)
        self.dtype         = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr   = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    # ------------------------------------------------------------------
    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros(
            (self.num_classes, self.max_size, *self.feature_shape),
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------
    def add(
        self,
        samples: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ) -> None:
        """Insert samples into per-class ring buffers.

        Args:
            samples: (N, *feature_shape) — can be a Tensor or ndarray.
            labels:  (N,) integer class labels.
        """
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        samples = np.asarray(samples, dtype=self.dtype)
        labels  = np.asarray(labels).astype(np.int32)

        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(len(labels)):
            lbl = int(labels[i])
            idx = int(self.ptr[lbl])
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl]   = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    # ------------------------------------------------------------------
    def sample(
        self,
        labels: torch.Tensor | np.ndarray,
        n_samples: int,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Sample stored entries for each label.

        Args:
            labels:    (B,) integer class labels.
            n_samples: Number of samples to draw per label.
            device:    Target torch device for the returned tensor.

        Returns:
            Tensor of shape (B, n_samples, *feature_shape).
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy().astype(np.int32)
        else:
            labels_np = np.asarray(labels).astype(np.int32)

        B = labels_np.shape[0]
        sample_indices = np.empty((B, n_samples), dtype=np.int32)

        for i in range(B):
            lbl   = int(labels_np[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros(n_samples, dtype=np.int32)
            else:
                choice = np.random.choice if rng is None else rng.choice
                sample_indices[i] = choice(
                    valid, n_samples, replace=(valid < n_samples)
                )

        # bank[labels, indices] => (B, n_samples, *feature_shape)
        out = self.bank[labels_np[:, None], sample_indices]
        t = torch.from_numpy(out.copy())
        if device is not None:
            t = t.to(device)
        return t

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.count.sum())

    def is_ready(self, min_per_class: int = 1) -> bool:
        """True if every class has at least `min_per_class` samples."""
        return bool((self.count >= min_per_class).all())

    def save_npz(self, path: str | Path) -> None:
        """Persist the bank without Python pickles.

        Historical generated replay uses one frozen per-rank snapshot.  Keeping
        it outside the model checkpoint avoids inflating every EMA checkpoint.
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("Cannot save an empty MemoryBank.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            bank=self.bank,
            ptr=self.ptr,
            count=self.count,
        )

    def load_npz(self, path: str | Path) -> None:
        """Restore a snapshot written by :meth:`save_npz`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as state:
            bank = np.asarray(state["bank"])
            ptr = np.asarray(state["ptr"], dtype=np.int32)
            count = np.asarray(state["count"], dtype=np.int32)
        expected_prefix = (self.num_classes, self.max_size)
        if bank.ndim < 2 or tuple(bank.shape[:2]) != expected_prefix:
            raise ValueError(
                f"Snapshot bank shape {bank.shape} does not match {expected_prefix}."
            )
        if ptr.shape != (self.num_classes,) or count.shape != (self.num_classes,):
            raise ValueError("Snapshot pointer/count shapes do not match num_classes.")
        self.bank = np.asarray(bank, dtype=self.dtype)
        self.feature_shape = tuple(self.bank.shape[2:])
        self.ptr = ptr.copy()
        self.count = count.copy()


__all__ = ["ArrayMemoryBank"]
