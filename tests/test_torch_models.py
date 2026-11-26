"""Cross-check: PyTorch's autograd should give exactly the gradients our NumPy backprop gives."""

import numpy as np
import torch
import torch.nn.functional as F

from mnist.numpy_mlp import NumpyMLP
from mnist.torch_models import SmallCNN, TorchMLP


def test_torch_mlp_matches_numpy_loss_and_gradients():
    rng = np.random.default_rng(0)
    X = rng.random((32, 784))
    y = rng.integers(0, 10, size=32)

    np_model = NumpyMLP(seed=0, dtype=np.float64)
    np_loss, np_grads = np_model.loss_and_grads(X, y)

    t_model = TorchMLP.from_numpy(np_model.params, dtype=torch.float64)
    t_loss = F.cross_entropy(t_model(torch.from_numpy(X)), torch.from_numpy(y))
    t_loss.backward()

    assert np.isclose(t_loss.item(), np_loss, rtol=1e-12)
    # nn.Linear stores weights as (out, in), so compare against the transpose
    pairs = [("W1", t_model.fc1.weight.grad.T), ("b1", t_model.fc1.bias.grad),
             ("W2", t_model.fc2.weight.grad.T), ("b2", t_model.fc2.bias.grad)]
    for name, t_grad in pairs:
        np.testing.assert_allclose(t_grad.numpy(), np_grads[name], rtol=1e-9, atol=1e-12)


def test_cnn_output_shape():
    model = SmallCNN()
    logits = model(torch.zeros(5, 784))
    assert logits.shape == (5, 10)
