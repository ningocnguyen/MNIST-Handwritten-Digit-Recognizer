"""Load the MNIST dataset straight from the original IDX binary files.

IDX format (big-endian):
    images: [magic=2051][count][rows][cols] then count*rows*cols unsigned bytes
    labels: [magic=2049][count]             then count unsigned bytes
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


def _read_idx_images(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    # ">u4" = big-endian unsigned 32-bit int; the header is 4 of them
    magic, count, rows, cols = np.frombuffer(raw, dtype=">u4", count=4)
    if magic != IMAGE_MAGIC:
        raise ValueError(f"{path.name}: expected magic {IMAGE_MAGIC}, got {magic}")
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
    return pixels.reshape(count, rows * cols)  # one flattened image per row


def _read_idx_labels(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, count = np.frombuffer(raw, dtype=">u4", count=2)
    if magic != LABEL_MAGIC:
        raise ValueError(f"{path.name}: expected magic {LABEL_MAGIC}, got {magic}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if labels.size != count:
        raise ValueError(f"{path.name}: header says {count} labels, found {labels.size}")
    return labels.astype(np.int64)


def load_split(prefix: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load one split ('train' or 't10k'). Returns X (N, 784) float32 in [0, 1], y (N,) int64."""
    X = _read_idx_images(data_dir / f"{prefix}-images.idx3-ubyte")
    y = _read_idx_labels(data_dir / f"{prefix}-labels.idx1-ubyte")
    if len(X) != len(y):
        raise ValueError(f"{prefix}: {len(X)} images but {len(y)} labels")
    return X.astype(np.float32) / 255.0, y


def load_mnist(val_size: int = 5000, seed: int = 0, data_dir: Path = DATA_DIR):
    """Return (X_train, y_train), (X_val, y_val), (X_test, y_test).

    The validation set is carved out of the 60k training images, so the
    10k test set stays untouched until the final evaluation.
    """
    X_full, y_full = load_split("train", data_dir)
    X_test, y_test = load_split("t10k", data_dir)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_full))
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    return (
        (X_full[train_idx], y_full[train_idx]),
        (X_full[val_idx], y_full[val_idx]),
        (X_test, y_test),
    )


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Turn labels like [3, 0] into rows like [0,0,0,1,0,...], [1,0,0,...]."""
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out
