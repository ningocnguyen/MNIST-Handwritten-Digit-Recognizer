"""PyTorch versions: the same MLP as the NumPy one (sanity check), then a small CNN.

    python -m mnist.torch_models --model mlp     # should land near the NumPy MLP (~97.7%)
    python -m mnist.torch_models --model cnn     # ~99%

Same data split, same early-stopping-on-val rule and single test evaluation as
mnist/train.py; the difference is that PyTorch computes the gradients (autograd)
instead of our hand-written backward().
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnist.data import load_mnist
from mnist.numpy_mlp import NumpyMLP
from mnist.plotting import plot_history

ROOT = Path(__file__).resolve().parent.parent


class TorchMLP(nn.Module):
    """784 -> 128 (ReLU) -> 10. Outputs raw scores (logits); softmax lives inside the loss."""

    def __init__(self, sizes=(784, 128, 10)):
        super().__init__()
        n_in, n_hidden, n_out = sizes
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    @classmethod
    def from_numpy(cls, params: dict, dtype=torch.float32):
        """Copy weights from a NumpyMLP. nn.Linear stores W as (out, in), ours is (in, out): transpose."""
        n_in, n_hidden = params["W1"].shape
        model = cls((n_in, n_hidden, params["W2"].shape[1])).to(dtype)
        with torch.no_grad():
            model.fc1.weight.copy_(torch.from_numpy(params["W1"].T.copy()))
            model.fc1.bias.copy_(torch.from_numpy(params["b1"]))
            model.fc2.weight.copy_(torch.from_numpy(params["W2"].T.copy()))
            model.fc2.bias.copy_(torch.from_numpy(params["b2"]))
        return model


class SmallCNN(nn.Module):
    """Two conv blocks, then a small classifier.

    (1, 28, 28) -conv3x3-> (32, 28, 28) -pool-> (32, 14, 14)
                -conv3x3-> (64, 14, 14) -pool-> (64, 7, 7) -> 3136 -> 128 -> 10
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # flat 784 pixels back into a 28x28 image
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, X, y, batch_size: int = 5000):
    """Mean cross-entropy and accuracy. eval() turns dropout off."""
    model.eval()
    total_loss, correct = 0.0, 0
    for start in range(0, len(X), batch_size):
        logits = model(X[start:start + batch_size])
        yb = y[start:start + batch_size]
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def fit(model, optimizer, train_data, val_data, *, epochs, batch_size, patience, seed=0,
        verbose=True):
    """Same loop as mnist.train.train, with autograd + an optimizer doing the update."""
    X_train, y_train = train_data
    rng = np.random.default_rng(seed)  # same shuffling as the NumPy loop

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        model.train()  # dropout on
        order = torch.from_numpy(rng.permutation(len(X_train))).to(X_train.device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
            optimizer.zero_grad()  # PyTorch accumulates gradients, so clear last step's
            loss.backward()  # autograd: replaces our hand-written backward()
            optimizer.step()  # replaces sgd_step()

        train_loss, train_acc = evaluate(model, *train_data)
        val_loss, val_acc = evaluate(model, *val_data)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"epoch {epoch:2d}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}  "
                  f"({time.perf_counter() - start_time:.1f}s)" + ("  *" if improved else ""))

        if epoch - best_epoch >= patience:
            if verbose:
                print(f"early stop: no val improvement for {patience} epochs")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history


# Per-model defaults. The MLP copies the NumPy setup exactly (plain SGD, lr 0.1) so the
# two results are directly comparable; the CNN uses Adam, the usual default for CNNs.
DEFAULTS = {
    "mlp": dict(optimizer="sgd", lr=0.1, epochs=50, patience=5, device="cpu"),
    "cnn": dict(optimizer="adam", lr=1e-3, epochs=20, patience=3, device="auto"),
}
TITLES = {"mlp": "PyTorch MLP (784-128-10) on MNIST", "cnn": "PyTorch CNN on MNIST"}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple-silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--optimizer", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", help="cpu, mps, cuda or auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for key, value in DEFAULTS[args.model].items():  # fill unset options with per-model defaults
        if getattr(args, key) is None:
            setattr(args, key, value)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_np, val_np, test_np = load_mnist(seed=args.seed)
    to_t = lambda X, y: (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))  # noqa: E731
    train_data, val_data, (X_test, y_test) = to_t(*train_np), to_t(*val_np), to_t(*test_np)

    if args.model == "mlp":
        # Start from the exact initial weights the NumPy MLP used (same seed).
        model = TorchMLP.from_numpy(NumpyMLP(seed=args.seed).params)
    else:
        model = SmallCNN()
    model.to(device)

    opt_cls = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=args.lr)
    print(f"{args.model}: {count_params(model):,} parameters, {args.optimizer} lr {args.lr}, "
          f"device {device}")

    history = fit(model, optimizer, train_data, val_data, epochs=args.epochs,
                  batch_size=args.batch_size, patience=args.patience, seed=args.seed)

    # The one and only look at the test set.
    test_loss, test_acc = evaluate(model, X_test, y_test)
    best = history["best_epoch"]
    print(f"\nbest epoch {best}: val acc {history['val_acc'][best - 1]:.4f}")
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}")

    name = f"torch_{args.model}"
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / f"{name}.pt")
    summary = {"args": vars(args), "params": count_params(model), "test_loss": test_loss,
               "test_acc": test_acc, **history}
    (models_dir / f"{name}_history.json").write_text(json.dumps(summary, indent=2))
    plot_path = ROOT / "docs" / f"{name}_curves.png"
    plot_history(history, plot_path, title=TITLES[args.model])
    print(f"saved models/{name}.pt, models/{name}_history.json, {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
