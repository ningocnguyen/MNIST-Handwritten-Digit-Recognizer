"""Draw a digit and watch all three models classify it.

    python app.py        # then open the printed URL (http://127.0.0.1:7860)

Needs the trained weights in models/ (see README: python -m mnist.train, etc.).
"""

import gradio as gr
import numpy as np

from mnist.inference import load_models
from mnist.preprocess import canvas_to_ink, to_mnist

_loaded = load_models()
MODELS = {  # display order: best model first
    "PyTorch CNN": _loaded["PyTorch CNN"],
    "PyTorch MLP": _loaded["PyTorch MLP"],
    "NumPy MLP (from scratch)": _loaded["NumPy MLP"],
}


def classify(canvas):
    """Sketchpad value -> (28x28 preview, one {digit: probability} dict per model)."""
    image = canvas.get("composite") if isinstance(canvas, dict) else canvas
    x = None if image is None else to_mnist(canvas_to_ink(np.asarray(image)))
    if x is None:
        return None, *[None] * len(MODELS)

    preview = np.kron(x.reshape(28, 28), np.ones((8, 8)))  # 28x28 -> 224x224, blocky on purpose
    preview = (255 * (1 - preview)).astype(np.uint8)  # black ink on white, like the canvas
    outputs = [{str(d): float(p) for d, p in enumerate(fn(x[None])[0])} for fn in MODELS.values()]
    return preview, *outputs


with gr.Blocks(title="MNIST digit recognizer") as demo:
    gr.Markdown(
        "# Draw a digit\n"
        "Three models classify it as you draw: a CNN, an MLP in PyTorch, and the same MLP "
        "written from scratch in NumPy. *What the model sees* shows your drawing after "
        "MNIST-style preprocessing (cropped, scaled to 20×20, centred in 28×28)."
    )
    with gr.Row():
        with gr.Column(scale=3):
            canvas = gr.Sketchpad(
                label="Draw here",
                canvas_size=(280, 280),
                fixed_canvas=True,
                image_mode="RGBA",
                type="numpy",
                brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=18),
                layers=False,
            )
            preview = gr.Image(label="What the model sees (28×28)", height=224, width=224,
                               interactive=False)
        with gr.Column(scale=2):
            labels = [gr.Label(label=name, num_top_classes=3) for name in MODELS]

    canvas.change(classify, inputs=canvas, outputs=[preview, *labels])

if __name__ == "__main__":
    demo.launch()
