"""Tests for the canvas -> MNIST preprocessing used by the demo app."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mnist.preprocess import canvas_to_ink, to_mnist

ROOT = Path(__file__).resolve().parent.parent


def test_empty_canvas_returns_none():
    assert to_mnist(np.zeros((280, 280))) is None


def test_output_is_mnist_shaped_and_centred():
    ink = np.zeros((400, 400))
    ink[30:150, 300:320] = 1.0  # a small vertical stroke in the top-right corner
    x = to_mnist(ink)
    assert x.shape == (784,) and x.dtype == np.float32
    assert np.isclose(x.max(), 1.0)
    img = x.reshape(28, 28)
    ys, xs = np.indices(img.shape)
    cy, cx = (ys * img).sum() / img.sum(), (xs * img).sum() / img.sum()
    assert abs(cy - 14) < 1 and abs(cx - 14) < 1  # moved to the centre
    assert img[:4].sum() == 0 and img[-4:].sum() == 0  # 20x20 box leaves a 4px border


def test_canvas_to_ink_handles_transparent_and_white_backgrounds():
    transparent = np.zeros((2, 2, 4), dtype=np.uint8)  # empty = fully transparent
    transparent[0, 0] = [0, 0, 0, 255]  # one black, opaque pixel
    white = np.full((2, 2, 4), 255, dtype=np.uint8)
    white[0, 0] = [0, 0, 0, 255]
    for canvas in (transparent, white):
        ink = canvas_to_ink(canvas)
        assert ink[0, 0] == 1.0 and ink[1, 1] == 0.0


@pytest.mark.skipif(not (ROOT / "models/torch_cnn.pt").exists() or not (ROOT / "data").exists(),
                    reason="needs data/ and trained models/")
def test_cnn_still_correct_on_redrawn_test_digits():
    """Simulate drawing: enlarge real test digits 10x onto a big canvas, off-centre.

    After preprocessing, the CNN should still recognise almost all of them.
    """
    import torch

    from mnist.data import load_split
    from mnist.torch_models import SmallCNN

    X, y = load_split("t10k")
    model = SmallCNN()
    model.load_state_dict(torch.load(ROOT / "models/torch_cnn.pt", map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), 200, replace=False)
    inputs = []
    for i in idx:
        big = Image.fromarray((X[i].reshape(28, 28) * 255).astype(np.uint8)).resize((280, 280))
        canvas = np.zeros((500, 500), dtype=np.float32)
        top, left = rng.integers(0, 220, size=2)
        canvas[top:top + 280, left:left + 280] = np.asarray(big) / 255.0
        inputs.append(to_mnist(canvas))
    with torch.no_grad():
        pred = model(torch.from_numpy(np.stack(inputs))).argmax(1).numpy()
    assert (pred == y[idx]).mean() >= 0.97
