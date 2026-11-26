"""Turn a hand-drawn canvas image into an MNIST-style 28x28 input.

The MNIST digits were made this way (LeCun et al.): each digit was scaled to fit
a 20x20 box, keeping its aspect ratio, then placed in a 28x28 frame so its
centre of mass sits in the middle. A drawing has to go through the same steps,
otherwise a small or off-centre digit looks nothing like the training data.
"""

import numpy as np
from PIL import Image

INK_THRESHOLD = 0.1


def canvas_to_ink(rgba: np.ndarray) -> np.ndarray:
    """(H, W, 4) uint8 RGBA -> (H, W) float 'ink' in [0, 1].

    Dark, opaque pixels are ink. Works for a transparent background as well as
    an opaque white one.
    """
    rgba = rgba.astype(np.float32) / 255.0
    if rgba.ndim == 2:  # already grayscale: dark = ink
        return 1.0 - rgba
    gray = rgba[..., :3].mean(axis=2)
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else 1.0
    return alpha * (1.0 - gray)


def to_mnist(ink: np.ndarray) -> np.ndarray | None:
    """(H, W) ink in [0, 1] -> (784,) float32 MNIST-style input, or None if nothing is drawn."""
    mask = ink > INK_THRESHOLD
    if not mask.any():
        return None

    # 1. crop to the bounding box of the ink
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    digit = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # 2. pad to a square so resizing keeps the aspect ratio (a "1" stays thin)
    h, w = digit.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = digit

    # 3. shrink to 20x20 with anti-aliasing (soft edges, like the MNIST scans)
    small = Image.fromarray((square * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    small = np.asarray(small, dtype=np.float32) / 255.0

    # 4. place in 28x28 so the centre of mass lands at the centre (14, 14)
    ys, xs = np.indices(small.shape)
    total = small.sum()
    cy, cx = (ys * small).sum() / total, (xs * small).sum() / total
    top = int(np.clip(round(14 - cy), 0, 8))  # clip keeps the 20x20 box inside the frame
    left = int(np.clip(round(14 - cx), 0, 8))
    out = np.zeros((28, 28), dtype=np.float32)
    out[top:top + 20, left:left + 20] = small

    return (out / out.max()).reshape(784)  # strongest stroke = 1.0, like the training data
