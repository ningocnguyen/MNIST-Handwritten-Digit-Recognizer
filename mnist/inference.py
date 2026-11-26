"""Load the three trained models for prediction (used by the notebook and the demo app)."""

from pathlib import Path

import numpy as np
import torch

from mnist.numpy_mlp import NumpyMLP
from mnist.torch_models import SmallCNN, TorchMLP

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_models(models_dir: Path = MODELS_DIR) -> dict:
    """Return {name: predict}, where predict maps (N, 784) float32 pixels to (N, 10) probabilities."""
    numpy_mlp = NumpyMLP()
    numpy_mlp.params = dict(np.load(models_dir / "numpy_mlp.npz"))

    def numpy_predict(X):
        return np.exp(numpy_mlp.forward(X)[0])

    def torch_predictor(model, weights):
        model.load_state_dict(torch.load(models_dir / weights, map_location="cpu"))
        model.eval()  # dropout off

        @torch.no_grad()
        def predict(X):
            return torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()

        return predict

    return {
        "NumPy MLP": numpy_predict,
        "PyTorch MLP": torch_predictor(TorchMLP(), "torch_mlp.pt"),
        "PyTorch CNN": torch_predictor(SmallCNN(), "torch_cnn.pt"),
    }
