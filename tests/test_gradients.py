"""Check the hand-written backprop against numerical gradients.

For every parameter entry w, the centered difference

    (L(w + eps) - L(w - eps)) / (2 * eps)

approximates dL/dw independently of the backprop code. If the two agree to a
relative error of about 1e-7, the backprop math is right.

Uses float64 (in float32, rounding noise swamps the tiny loss differences) and
a small network so checking every single entry stays fast.
"""

import numpy as np
import pytest

from mnist.numpy_mlp import NumpyMLP


def numerical_grad(model, X, y, name, eps=1e-5):
    param = model.params[name]
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        original = param[idx]
        param[idx] = original + eps
        loss_plus = model.loss(model.forward(X)[0], y)
        param[idx] = original - eps
        loss_minus = model.loss(model.forward(X)[0], y)
        param[idx] = original  # put it back
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(0)
    X = rng.random((8, 20))  # 8 fake "images" with 20 pixels each
    y = rng.integers(0, 5, size=8)  # 5 classes
    model = NumpyMLP(sizes=(20, 12, 5), seed=0, dtype=np.float64)
    return model, X, y


@pytest.mark.parametrize("name", ["W1", "b1", "W2", "b2"])
def test_backprop_matches_numerical_gradient(small_problem, name):
    model, X, y = small_problem
    _, grads = model.loss_and_grads(X, y)
    num = numerical_grad(model, X, y, name)
    assert grads[name].shape == model.params[name].shape
    assert relative_error(grads[name], num) < 1e-6


def test_initial_loss_is_about_log_num_classes():
    # Untrained, the network should spread probability roughly evenly over the
    # 10 classes, so the loss starts near -log(1/10) = log(10) ~ 2.30.
    rng = np.random.default_rng(1)
    X = rng.random((256, 784)).astype(np.float32) * 0.3
    y = rng.integers(0, 10, size=256)
    model = NumpyMLP(seed=0)
    loss = model.loss(model.forward(X)[0], y)
    assert abs(loss - np.log(10)) < 0.5


def test_full_size_shapes_and_dtype():
    model = NumpyMLP(seed=0)
    X = np.zeros((4, 784), dtype=np.float32)
    y = np.array([0, 1, 2, 3])
    log_probs, _ = model.forward(X)
    assert log_probs.shape == (4, 10)
    assert np.allclose(np.exp(log_probs).sum(axis=1), 1.0, atol=1e-5)
    _, grads = model.loss_and_grads(X, y)
    for name, g in grads.items():
        assert g.shape == model.params[name].shape
        assert g.dtype == np.float32
