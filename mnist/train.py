"""Train the NumPy MLP with mini-batch SGD and early stopping on the validation set.

    python -m mnist.train                  # defaults: lr 0.1, batch 64, up to 50 epochs
    python -m mnist.train --lr 0.05 --patience 3

Flow:
  1. every epoch: shuffle the 55k training images, take SGD steps on mini-batches
  2. after each epoch: measure loss/accuracy on train and on the 5k validation set
  3. keep a copy of the weights from the best validation epoch; stop once validation
     accuracy hasn't improved for `patience` epochs
  4. restore the best weights, then evaluate on the 10k test set exactly once
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mnist.data import load_mnist
from mnist.numpy_mlp import NumpyMLP
from mnist.plotting import plot_history

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model: NumpyMLP, X: np.ndarray, y: np.ndarray, batch_size: int = 10_000):
    """Mean cross-entropy loss and accuracy over a whole dataset (in chunks to bound memory)."""
    total_loss, correct = 0.0, 0
    for start in range(0, len(X), batch_size):
        Xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        log_probs, _ = model.forward(Xb)
        total_loss += model.loss(log_probs, yb) * len(yb)
        correct += int((log_probs.argmax(axis=1) == yb).sum())
    return total_loss / len(X), correct / len(X)


def sgd_step(model: NumpyMLP, grads: dict, lr: float) -> None:
    """Plain SGD: w <- w - lr * dL/dw (in place)."""
    for name, g in grads.items():
        model.params[name] -= lr * g


def train(model, train_data, val_data, *, epochs=50, batch_size=64, lr=0.1, patience=5,
          seed=0, verbose=True):
    """Train in place. Returns a history dict; the model ends with its best-validation weights."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    rng = np.random.default_rng(seed)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch, best_params = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_train))  # new random order every epoch
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            _, grads = model.loss_and_grads(X_train[idx], y_train[idx])
            sgd_step(model, grads, lr)

        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            best_params = {k: v.copy() for k, v in model.params.items()}

        if verbose:
            print(f"epoch {epoch:2d}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}" + ("  *" if improved else ""))

        if epoch - best_epoch >= patience:
            if verbose:
                print(f"early stop: no val improvement for {patience} epochs")
            break

    model.params = best_params
    history["best_epoch"] = best_epoch
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_data, val_data, (X_test, y_test) = load_mnist(seed=args.seed)
    model = NumpyMLP(seed=args.seed)
    history = train(model, train_data, val_data, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                    seed=args.seed)

    # The one and only look at the test set.
    test_loss, test_acc = evaluate(model, X_test, y_test)
    best = history["best_epoch"]
    print(f"\nbest epoch {best}: val acc {history['val_acc'][best - 1]:.4f}")
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    np.savez(models_dir / "numpy_mlp.npz", **model.params)
    summary = {"args": vars(args), "test_loss": test_loss, "test_acc": test_acc, **history}
    (models_dir / "numpy_mlp_history.json").write_text(json.dumps(summary, indent=2))

    plot_path = ROOT / "docs" / "numpy_mlp_curves.png"
    plot_history(history, plot_path, title="NumPy MLP (784-128-10) on MNIST")
    print(f"saved models/numpy_mlp.npz, models/numpy_mlp_history.json, {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
