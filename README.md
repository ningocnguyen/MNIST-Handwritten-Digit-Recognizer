# MNIST from Scratch (Python)

A handwritten-digit classifier built twice: once from scratch in NumPy (hand-written backprop), once in PyTorch. A Python rebuild of my earlier Java project.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the MNIST IDX files (e.g. from Kaggle "MNIST Dataset") and put these four in `data/`:

```
train-images.idx3-ubyte  train-labels.idx1-ubyte
t10k-images.idx3-ubyte   t10k-labels.idx1-ubyte
```

Check the data loads:

```bash
python scripts/show_samples.py
```

## Status

- [x] Step 1: data loader
- [x] Step 2: NumPy MLP + gradient check
- [x] Step 3: training + curves
- [x] Step 4: PyTorch MLP + CNN
- [x] Step 5: comparison
- [ ] Step 6: draw-a-digit demo
