#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch==2.7.0",
#     "numpy==1.26.4",
#     "onnx==1.17.0",
#     "onnxruntime==1.22.0",
#     "huggingface_hub",
#     "coremltools==9.0; sys_platform == 'darwin'",
# ]
# ///
"""
Exports Smooth Motion's two networks to ONNX for `vergence-worker` (the ONNX
Runtime worker for Windows and Linux PCs), at the same fixed sizes and with
the same inputs and outputs as the Core ML conversions `vergence worker` and
`vergence analyze` run:

  RIFE 4.25 lite at 480x288, block scales 8 4 2 1 1, timestep an input (½)
    inputs  img0, img1: [1,3,288,480] float32, RGB 0…1;  timestep: [1,1,1,1]
    outputs flow: [1,4,288,480] (F½→0 x,y then F½→1 x,y, pixels), mask: [1,1,288,480]
  NeuFlow v2 at 512x288, one way (img0→img1), 4 refinement iterations, fp32
    inputs  img0, img1: [1,3,288,512] float32, RGB 0…255
    outputs flow: [1,2,288,512] (x, y, pixels)

    uv run scripts/export-onnx-models.py [--hypnos ../Hypnos] [--out onnx-models] [--coreml DIR]

The network code is not repeated here: it is imported from Hypnos's
`scripts/convert-interp-model.py` (its RIFE 4.25 IFNet re-implementation and
its NeuFlow loading, rank-5 upsample and one-way wrapper), so the ONNX and the
Core ML models are traced from one definition. The weights are the same ones
it converts, from its download cache:

  <hypnos>/build/interp-spike/dl/rife425lite/train_log/flownet.pkl
      RIFE 4.25 lite, hzwer/Practical-RIFE — MIT
  <hypnos>/build/interp-spike/NeuFlow_v2/ (clone with neuflow_mixed.pth)
      neufieldrobotics/NeuFlow_v2 — Apache-2.0 (its package code is imported)

Override either with --rife-weights / --neuflow-repo. The .onnx files are
derived from those weights: keep them out of git (`/onnx-models/` is ignored)
and ship them under the licences above.

Each export is checked against PyTorch with ONNX Runtime (CPU), and, on a Mac
with --coreml DIR, against the shipping Core ML conversion (fed float16 the way
StereoFlowAnalyzer feeds it), on the converter's synthetic pan and on noise.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

RIFE_SIZE = (480, 288)
RIFE_SCALES = [8, 4, 2, 1, 1]
NEUFLOW_SIZE = (512, 288)
NEUFLOW_ITERS = 4
RIFE_NAME = "RIFE425lite_480x288"
NEUFLOW_NAME = "NeuFlowV2_512x288_1way_i4_fp32"


def load_converter(hypnos):
    path = Path(hypnos) / "scripts" / "convert-interp-model.py"
    if not path.exists():
        sys.exit(f"{path} not found: pass --hypnos (a Hypnos checkout)")
    # No __pycache__ in the Hypnos checkout (or the NeuFlow clone it imports).
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("convert_interp_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(module, example, input_names, output_names, path):
    with torch.no_grad():
        torch.onnx.export(module, example, str(path), input_names=input_names, output_names=output_names,
                          opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    onnx.checker.check_model(str(path))
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


def report(label, got, ref):
    err = np.abs(np.asarray(got, np.float32) - np.asarray(ref, np.float32))
    print(f"  {label}: max |Δ| {err.max():.4f}, mean {err.mean():.5f}, p99 {np.percentile(err, 99):.4f}")


def check(path, feeds, reference, output_names, coreml_path):
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for case, feed in feeds.items():
        got = session.run(output_names, {k: v.numpy() for k, v in feed.items()})
        for name, g, r in zip(output_names, got, reference[case]):
            report(f"{case} {name}: ONNX Runtime vs PyTorch", g, r.numpy())
        if coreml_path is not None and coreml_path.exists():
            import coremltools as ct
            ml = ct.models.MLModel(str(coreml_path))
            cm = ml.predict({k: v.numpy().astype(np.float16) for k, v in feed.items()})
            for name, g in zip(output_names, got):
                report(f"{case} {name}: ONNX Runtime vs Core ML", g, cm[name])


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--hypnos", default=str(root.parent / "Hypnos"), help="a Hypnos checkout")
    ap.add_argument("--rife-weights", help="RIFE 4.25 lite train_log dir (flownet.pkl)")
    ap.add_argument("--neuflow-repo", help="NeuFlow_v2 clone with neuflow_mixed.pth")
    ap.add_argument("--out", default=str(root / "onnx-models"))
    ap.add_argument("--coreml", help="folder of the Core ML conversions, to compare against")
    ap.add_argument("--only", choices=["rife", "neuflow"])
    args = ap.parse_args()

    conv = load_converter(args.hypnos)
    spike = Path(args.hypnos) / "build" / "interp-spike"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coreml = Path(args.coreml) if args.coreml else None
    torch.manual_seed(0)

    if args.only in (None, "rife"):
        w, h = RIFE_SIZE
        net = conv.load_rife(args.rife_weights or spike / "dl" / "rife425lite" / "train_log", RIFE_SCALES)
        a, b = conv.test_pair(h, w, scale255=False)
        ts = torch.full((1, 1, 1, 1), 0.5)
        noise = (torch.rand(1, 3, h, w), torch.rand(1, 3, h, w), ts)
        path = out / f"{RIFE_NAME}.onnx"
        export(net, (a, b, ts), ["img0", "img1", "timestep"], ["flow", "mask"], path)
        cases = {"pan": (a, b, ts), "noise": noise}
        with torch.no_grad():
            reference = {k: net(*v) for k, v in cases.items()}
        feeds = {k: dict(zip(["img0", "img1", "timestep"], v)) for k, v in cases.items()}
        check(path, feeds, reference, ["flow", "mask"], coreml / f"{RIFE_NAME}.mlpackage" if coreml else None)

    if args.only in (None, "neuflow"):
        w, h = NEUFLOW_SIZE
        net, up_mod = conv.load_neuflow(args.neuflow_repo or spike / "NeuFlow_v2", h, w)
        net.init_bhwd(1, h, w, "cpu", amp=False)
        up_mod.UpSample.forward = conv.rank5_upsample_forward
        model = conv.NeuFlowBidirectional(net, NEUFLOW_ITERS, one_way=True)

        class Copying(torch.nn.Module):
            """NeuFlow divides its inputs in place; trace on copies (as the Core ML conversion does)."""

            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, y):
                return self.m(x * 1.0, y * 1.0)

        model = Copying(model).eval()
        a, b = conv.test_pair(h, w, scale255=True)
        noise = (torch.rand(1, 3, h, w) * 255, torch.rand(1, 3, h, w) * 255)
        path = out / f"{NEUFLOW_NAME}.onnx"
        export(model, (a.clone(), b.clone()), ["img0", "img1"], ["flow"], path)
        cases = {"pan": (a, b), "noise": noise}
        with torch.no_grad():
            reference = {k: (model(v[0].clone(), v[1].clone()),) for k, v in cases.items()}
        feeds = {k: dict(zip(["img0", "img1"], v)) for k, v in cases.items()}
        check(path, feeds, reference, ["flow"], coreml / f"{NEUFLOW_NAME}.mlpackage" if coreml else None)


if __name__ == "__main__":
    main()
