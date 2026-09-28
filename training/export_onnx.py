"""Export a trained checkpoint to ONNX for onnxruntime-web and verify it.

  python export_onnx.py --run runs/transformer --out export/

Produces:
  model.onnx    input  "landmarks" float32 [1, 30, 147]
                output "probs"     float32 [1, num_classes]  (softmax already applied)
  labels.json   class names in output order + feature/normalisation metadata
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

import config as C
from evaluate import load_checkpoint


class WithSoftmax(nn.Module):
    """Wrap the classifier so the browser receives probabilities directly."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)


def export(model: nn.Module, dummy: torch.Tensor, path: Path, opset: int, dynamic_batch: bool):
    kw = dict(input_names=["landmarks"], output_names=["probs"], opset_version=opset)
    if dynamic_batch:
        kw["dynamic_axes"] = {"landmarks": {0: "batch"}, "probs": {0: "batch"}}
    try:
        # TorchScript-based exporter: produces compact, widely-supported graphs (good for ORT-web).
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kw)
        return "torchscript"
    except Exception as e:  # legacy exporter unavailable in some torch versions
        print(f"[warn] legacy exporter failed ({type(e).__name__}: {e}); trying dynamo exporter")
        if dynamic_batch:
            kw.pop("dynamic_axes")
            kw["dynamic_shapes"] = ({0: torch.export.Dim("batch")},)
        torch.onnx.export(model, (dummy,), str(path), dynamo=True, **kw)
        return "dynamo"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run dir containing best.pt")
    ap.add_argument("--out", type=Path, default=C.EXPORT_DIR)
    ap.add_argument("--opset", type=int, default=C.ONNX_OPSET)
    ap.add_argument("--dynamic-batch", action="store_true", help="export with a dynamic batch axis")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.run / "best.pt")
    wrapped = WithSoftmax(model).eval()
    seq_len, feat_dim = ckpt["seq_len"], ckpt["feat_dim"]
    dummy = torch.randn(1, seq_len, feat_dim)

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "model.onnx"
    exporter = export(wrapped, dummy, onnx_path, args.opset, args.dynamic_batch)

    # Re-save as a single self-contained file (the web app fetches one URL).
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_path), save_as_external_data=False)
    stale = args.out / "model.onnx.data"
    if stale.exists():
        stale.unlink()

    # Verify numerically against PyTorch with onnxruntime.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(5):
        x = rng.normal(0, 0.5, (1, seq_len, feat_dim)).astype(np.float32)
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(x)).numpy()
        got = sess.run(["probs"], {"landmarks": x})[0]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
    ok = max_diff < 1e-4
    ins, outs = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"[onnx] exporter={exporter} opset={args.opset} size={onnx_path.stat().st_size / 1e6:.2f} MB")
    print(f"[onnx] input {ins.name} {ins.shape} -> output {outs.name} {outs.shape}")
    print(f"[verify] max |torch - onnxruntime| = {max_diff:.2e} -> {'PASS' if ok else 'FAIL'}")

    meta = {
        "labels": ckpt["labels"],
        "model": ckpt["model_name"],
        "input_name": "landmarks", "output_name": "probs", "output_is_softmax": True,
        "seq_len": seq_len, "feat_dim": feat_dim,
        "feature_layout": {"left_hand": [C.LH_SLICE.start, C.LH_SLICE.stop],
                           "right_hand": [C.RH_SLICE.start, C.RH_SLICE.stop],
                           "pose": [C.POSE_SLICE.start, C.POSE_SLICE.stop] if feat_dim > C.POSE_SLICE.start else None,
                           "pose_mediapipe_ids": C.POSE_LANDMARK_IDS, "pose_names": C.POSE_NAMES,
                           "coords": "x,y,z per landmark, row-major"},
        "normalization": ("per frame: centre = shoulder midpoint (else wrist of first present hand); "
                          "scale = shoulder width in x/y (else max x/y extent of present hands); "
                          "(p - centre) / scale applied to x,y,z of present parts; missing parts = 0"),
        "resample": f"linear in time to {seq_len} frames",
        "val_top1": ckpt.get("val_top1"),
    }
    (args.out / "labels.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {onnx_path} and {args.out / 'labels.json'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
