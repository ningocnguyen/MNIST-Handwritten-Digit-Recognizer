"""Sanity check: load the data, print shapes, and save a grid of sample digits."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to a file; no window needed
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mnist.data import load_mnist  # noqa: E402

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

print(f"train: X {X_train.shape}  y {y_train.shape}")
print(f"val:   X {X_val.shape}  y {y_val.shape}")
print(f"test:  X {X_test.shape}  y {y_test.shape}")
print(f"pixel range: {X_train.min():.1f} to {X_train.max():.1f}  dtype {X_train.dtype}")
print("train label counts:", np.bincount(y_train))

fig, axes = plt.subplots(3, 8, figsize=(10, 4))
for ax, img, label in zip(axes.flat, X_train, y_train):
    ax.imshow(img.reshape(28, 28), cmap="gray_r")
    ax.set_title(str(label))
    ax.axis("off")
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "samples.png"
fig.savefig(out, dpi=100)
print(f"saved {out}")
