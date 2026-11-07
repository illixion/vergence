#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch==2.7.0",
#     "numpy==1.26.4",
#     "onnx==1.17.0",
#     "onnxruntime==1.22.0",
#     "transformers==4.51.3",
#     "pillow",
#     "huggingface_hub",
#     "coremltools==9.0; sys_platform == 'darwin'",
# ]
# ///
"""
Exports Depth Anything V2 Small to ONNX for `vergence-convert` (the Windows
and Linux converter), at the size and with the inputs Apple's Core ML build
(`DepthAnythingV2SmallF16`, what `vergence convert` runs on a Mac) takes:

  input   pixel_values: [1,3,392,518] float32, RGB normalised as ImageNet
          ((v/255 - mean) / std), the frame fitted inside and centred, the
          rest black
  output  predicted_depth: [1,392,518] float32, relative inverse depth
          (near high). The Core ML build divides it by its maximum; the
          converter does the same, which changes nothing downstream (every
          step is relative to each frame's own range).

The same names as the Apache-2.0 export at
huggingface.co/onnx-community/depth-anything-v2-small, which is what
`vergence-convert --download-model` fetches, so either file works.

    uv run scripts/export-depth-onnx.py [--out onnx-models] [--coreml PATH.mlpackage] [--image FRAME.png]
                                        [--compare OTHER.onnx]

Weights: depth-anything/Depth-Anything-V2-Small-hf (Apache-2.0; the Base and
Large weights are CC-BY-NC and are not offered). The .onnx file is derived
from them: keep it out of git (`/onnx-models/` is ignored) and ship it under
that licence.

With --coreml (on a Mac) and --image, the export is compared with the Core ML
model on that picture, each map divided by its maximum, as the converter
reads them. --compare does the same against another ONNX file (e.g. the
onnx-community download).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

WIDTH, HEIGHT = 518, 392
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NAME = "DepthAnythingV2Small"


class Wrapped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


def letterboxed(path):
    """The picture fitted inside 518x392, centred on black: (RGB uint8 HxWx3)."""
    from PIL import Image
    image = Image.open(path).convert("RGB")
    scale = min(WIDTH / image.width, HEIGHT / image.height)
    w, h = round(image.width * scale), round(image.height * scale)
    canvas = Image.new("RGB", (WIDTH, HEIGHT))
    canvas.paste(image.resize((w, h), Image.BILINEAR), ((WIDTH - w) // 2, (HEIGHT - h) // 2))
    return canvas, np.asarray(canvas)


def normalised(rgb):
    x = (rgb.astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].copy()


def report(label, a, b):
    a = a / a.max()
    b = b / b.max()
    d = np.abs(a - b)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    print(f"  {label}: max |Δ| {d.max():.4f}, mean {d.mean():.5f}, p99 {np.percentile(d, 99):.4f}, correlation {corr:.5f}")


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", default=str(root / "onnx-models"))
    ap.add_argument("--coreml", help="the Core ML DepthAnythingV2SmallF16.mlpackage, to compare against (Mac)")
    ap.add_argument("--image", help="a picture to compare on")
    ap.add_argument("--compare", help="another ONNX export to compare against")
    args = ap.parse_args()

    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").eval()
    wrapped = Wrapped(model).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}.onnx"
    example = torch.rand(1, 3, HEIGHT, WIDTH)
    with torch.no_grad():
        torch.onnx.export(wrapped, (example,), str(path), input_names=["pixel_values"],
                          output_names=["predicted_depth"], opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    onnx.checker.check_model(str(path))
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = wrapped(example).numpy()
    got = session.run(["predicted_depth"], {"pixel_values": example.numpy()})[0]
    report("noise: ONNX Runtime vs PyTorch", got, reference)

    if args.image:
        canvas, rgb = letterboxed(args.image)
        got = session.run(["predicted_depth"], {"pixel_values": normalised(rgb)})[0][0]
        if args.compare:
            other = ort.InferenceSession(args.compare, providers=["CPUExecutionProvider"])
            theirs = other.run(["predicted_depth"], {"pixel_values": normalised(rgb)})[0][0]
            report(f"{Path(args.image).name}: this export vs {Path(args.compare).name}", got, theirs)
        if args.coreml:
            import coremltools as ct
            ml = ct.models.MLModel(args.coreml)
            cm = np.asarray(ml.predict({"image": canvas})["depth"], np.float32)
            report(f"{Path(args.image).name}: ONNX Runtime vs Core ML", got, cm)


if __name__ == "__main__":
    main()
