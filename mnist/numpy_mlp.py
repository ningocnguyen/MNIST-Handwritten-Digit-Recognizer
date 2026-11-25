"""A 784 -> 128 -> 10 MLP written from scratch in NumPy.

    X (N, 784) --[W1, b1]--> z1 --ReLU--> a1 (N, 128) --[W2, b2]--> z2 (N, 10) --softmax--> probs

Loss: mean cross-entropy over the mini-batch.

Everything works on a whole mini-batch at once: each row of X is one image,
so one matrix multiply (X @ W1) replaces the per-sample loops of the Java version.
"""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """log(softmax(z)) row by row, computed stably.

    Subtracting the row max doesn't change softmax but keeps exp() from
    overflowing; working in log space keeps log() away from log(0).
    """
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


class NumpyMLP:
    """Two-layer MLP: ReLU hidden layer, softmax output, cross-entropy loss.

    Parameters live in `self.params` (W1, b1, W2, b2) so the training loop and
    the gradient check can loop over them by name. `backward` returns a dict with
    the same keys.
    """

    def __init__(self, sizes=(784, 128, 10), seed: int = 0, dtype=np.float32):
        n_in, n_hidden, n_out = sizes
        rng = np.random.default_rng(seed)
        # He init: std = sqrt(2 / fan_in). ReLU zeroes about half its inputs, so the
        # factor 2 keeps activation variance roughly constant from layer to layer.
        self.params = {
            "W1": (rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)).astype(dtype),
            "b1": np.zeros(n_hidden, dtype=dtype),
            "W2": (rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)).astype(dtype),
            "b2": np.zeros(n_out, dtype=dtype),
        }

    def forward(self, X: np.ndarray):
        """Return (log_probs, cache). log_probs is (N, n_out); cache keeps what backward needs."""
        p = self.params
        z1 = X @ p["W1"] + p["b1"]  # (N, 128); b1 broadcasts across the batch
        a1 = relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]  # (N, 10) raw scores ("logits")
        log_probs = log_softmax(z2)
        return log_probs, (X, z1, a1, log_probs)

    @staticmethod
    def loss(log_probs: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy: average of -log(probability given to the true class)."""
        return float(-log_probs[np.arange(len(y)), y].mean())

    def backward(self, cache, y: np.ndarray) -> dict:
        """Backprop: gradients of the mean cross-entropy loss w.r.t. every parameter."""
        X, z1, a1, log_probs = cache
        N = len(y)
        p = self.params

        # Output layer. Softmax + cross-entropy has a very simple combined derivative:
        #   dL/dz2 = probs - one_hot(y)        (divided by N because the loss is a mean)
        dz2 = np.exp(log_probs)
        dz2[np.arange(N), y] -= 1
        dz2 /= N

        dW2 = a1.T @ dz2  # (128, N) @ (N, 10) -> (128, 10); sums over the batch
        db2 = dz2.sum(axis=0)

        # Hidden layer: send the error back through W2, then through ReLU,
        # whose derivative is 1 where z1 > 0 and 0 elsewhere.
        da1 = dz2 @ p["W2"].T  # (N, 10) @ (10, 128) -> (N, 128)
        dz1 = da1 * (z1 > 0)

        dW1 = X.T @ dz1  # (784, N) @ (N, 128) -> (784, 128)
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        log_probs, cache = self.forward(X)
        return self.loss(log_probs, y), self.backward(cache, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted digit per row (argmax of the scores)."""
        log_probs, _ = self.forward(X)
        return log_probs.argmax(axis=1)
