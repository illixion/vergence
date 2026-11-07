#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch==2.7.0",
#     "numpy==1.26.4",
#     "onnx==1.17.0",
#     "onnxruntime==1.22.0",
#     "pillow",
#     "coremltools==9.0; sys_platform == 'darwin'",
# ]
# ///
"""
Exports GMNet (Liao et al., "Learning Gain Map for Inverse Tone Mapping",
ICLR 2025, MIT, github.com/qtlark/GMNet) to ONNX for `vergence-worker`'s
`ai-hdr` workload, with the inputs and outputs of Convolution's Core ML
conversion (Hypnos `scripts/convert-gainmap-model.py`, whose wrapper is
imported, so both come from one definition):

  inputs  image      [1,3,H,W] float32, RGB 0…1 (the worker rounds it to bytes
                     first, as the Core ML model's pixel-buffer input does);
                     H and W free, used at the Core ML model's enumerated sizes
          thumbnail  [1,3,256,256] float32, the whole frame squeezed
  outputs gain       [1,1,H,W]: the linear multiplier per pixel, 1 up to 5
          headroom   [1]: the largest gain

    git clone --depth 1 https://github.com/qtlark/GMNet.git /tmp/GMNet
    uv run scripts/export-gmnet-onnx.py --repo /tmp/GMNet [--hypnos ../Hypnos] [--out onnx-models]
                                        [--coreml .../GainMapGMNet.mlpackage]

Writes GainMapGMNet.onnx (the real-world checkpoint), checked against PyTorch
with ONNX Runtime (CPU) at every enumerated size, and with --coreml (on a
Mac) against the Core ML conversion on the same picture. The weights are
GMNet's (MIT): keep the .onnx out of git (`/onnx-models/` is ignored).
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

NAME = "GainMapGMNet"


def load_converter(hypnos):
    path = Path(hypnos) / "scripts" / "convert-gainmap-model.py"
    if not path.exists():
        sys.exit(f"{path} not found: pass --hypnos (a Hypnos checkout)")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("convert_gainmap_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedAdaptiveAvgPool(torch.nn.Module):
    """`AdaptiveAvgPool2d` as explicit slices and means: ONNX has no adaptive
    pooling to sizes that don't divide the input (GMNet pools its thumbnail
    branch to 3×3). The same bins PyTorch uses: [⌊i·n/k⌋, ⌈(i+1)·n/k⌉)."""

    def __init__(self, size):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        kh, kw = self.size
        rows = []
        for i in range(kh):
            y0, y1 = (i * h) // kh, -((-(i + 1) * h) // kh)
            cells = []
            for j in range(kw):
                x0, x1 = (j * w) // kw, -((-(j + 1) * w) // kw)
                cells.append(x[:, :, y0:y1, x0:x1].mean(dim=(2, 3), keepdim=True))
            rows.append(torch.cat(cells, dim=3))
        return torch.cat(rows, dim=2)


def replace_adaptive_pools(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.AdaptiveAvgPool2d) and child.output_size not in (1, (1, 1)):
            setattr(module, name, FixedAdaptiveAvgPool(child.output_size))
        else:
            replace_adaptive_pools(child)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo", required=True, help="a checkout of github.com/qtlark/GMNet")
    ap.add_argument("--hypnos", default=str(root.parent / "Hypnos"))
    ap.add_argument("--out", default=str(root / "onnx-models"))
    ap.add_argument("--coreml", help="Convolution's GainMapGMNet.mlpackage, to compare against")
    args = ap.parse_args()

    conv = load_converter(args.hypnos)
    GMNet = conv.load_gmnet(args.repo)
    net = GMNet(in_nc=3, out_nc=1, nf=64, nb=16, act_type="relu").eval()
    state = torch.load(Path(args.repo) / "checkpoints" / "G_realworld.pth", map_location="cpu")
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    model = conv.Exported(net, conv.PEAK["realworld"]).eval()
    torch.manual_seed(0)
    probe = (torch.rand(1, 3, 576, 1024), torch.rand(1, 3, 256, 256))
    with torch.no_grad():
        before = model(*probe)[0]
        replace_adaptive_pools(model)
        after = model(*probe)[0]
    print(f"explicit pooling vs PyTorch's adaptive pooling: max |Δ gain| {float((before - after).abs().max()):.2e}")

    # ONNX has no exp2: the wrapper's 2^q becomes e^(q·ln 2), the same number.
    torch.exp2 = lambda x: torch.exp(x * math.log(2.0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}.onnx"
    w0, h0 = conv.SIZES[0]
    torch.manual_seed(0)
    example = (torch.rand(1, 3, h0, w0), torch.rand(1, 3, 256, 256))
    with torch.no_grad():
        torch.onnx.export(model, example, str(path), input_names=["image", "thumbnail"],
                          output_names=["gain", "headroom"], opset_version=17, do_constant_folding=True,
                          dynamic_axes={"image": {2: "height", 3: "width"}, "gain": {2: "height", 3: "width"}},
                          dynamo=False)
    import onnx
    onnx.checker.check_model(str(path))
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # A smooth picture with a bright patch: something with highlights to expand.
    for w, h in conv.SIZES:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        picture = np.stack([xx / w, yy / h, 0.5 + 0.5 * np.sin(xx / 37) * np.cos(yy / 23)])
        picture[:, h // 3:h // 2, w // 3:w // 2] = 0.98
        picture = np.round(np.clip(picture, 0, 1) * 255) / 255
        thumb = torch.nn.functional.interpolate(torch.from_numpy(picture)[None], size=(256, 256), mode="area")
        thumb = torch.round(thumb * 255) / 255
        image = torch.from_numpy(picture)[None]
        with torch.no_grad():
            ref = model(image, thumb)[0].numpy()
        got = session.run(["gain"], {"image": image.numpy(), "thumbnail": thumb.numpy()})[0]
        d = np.abs(got - ref)
        print(f"  {w}x{h}: ONNX Runtime vs PyTorch: gain {ref.min():.3f}…{ref.max():.3f}, max |Δ| {d.max():.4f}, "
              f"mean {d.mean():.5f}")
        if args.coreml and (w, h) == conv.SIZES[0]:
            import coremltools as ct
            from PIL import Image
            ml = ct.models.MLModel(args.coreml)
            to_image = lambda a: Image.fromarray((np.transpose(a, (1, 2, 0)) * 255).round().astype(np.uint8))
            pred = ml.predict({"image": to_image(picture), "thumbnail": to_image(thumb[0].numpy())})
            cm = np.asarray(pred["gain"], dtype=np.float32)
            d = np.abs(np.log2(cm.reshape(got.shape)) - np.log2(got))
            print(f"  {w}x{h}: ONNX Runtime vs Core ML: log2 gain max |Δ| {d.max():.4f}, mean {d.mean():.5f}")


if __name__ == "__main__":
    main()
