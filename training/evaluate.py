"""Evaluate a trained checkpoint: top-1/top-5, per-class P/R/F1, confusion matrix PNG.

  python evaluate.py --run runs/transformer --data data/include50.npz
  python evaluate.py --run runs/transformer --data data/other.npz --split all

By default the test indices saved by train.py (splits.npz) are used.
Outputs (in the run dir): eval_<split>.json, confusion_<split>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from model import build_model
from train import load_npz


def load_checkpoint(path: Path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], ckpt["feat_dim"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


@torch.no_grad()
def predict(model, X, batch=256, device="cpu"):
    outs = [model(torch.from_numpy(X[i:i + batch]).to(device)).cpu() for i in range(0, len(X), batch)]
    return torch.cat(outs).numpy()


def plot_confusion(cm, labels, path, title):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(labels)
    size = max(6, 0.28 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised recall")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run dir containing best.pt")
    ap.add_argument("--data", type=Path, default=None, help="defaults to the dataset used in training")
    ap.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.run / "best.pt")
    labels = ckpt["labels"]
    d = load_npz(args.data or Path(ckpt["data"]))
    if d["labels"] != labels:
        raise SystemExit("Label list in the dataset differs from the checkpoint's labels.")
    idx = np.arange(len(d["y"])) if args.split == "all" else np.load(args.run / "splits.npz")[args.split]
    X, y = d["X"][idx].astype(np.float32), d["y"][idx]

    logits = predict(model, X)
    order = np.argsort(-logits, axis=1)
    top1 = float((order[:, 0] == y).mean())
    top5 = float((order[:, :5] == y[:, None]).any(1).mean())
    k = len(labels)
    p, r, f1, sup = precision_recall_fscore_support(y, order[:, 0], labels=range(k), zero_division=0)
    cm = confusion_matrix(y, order[:, 0], labels=range(k))

    per_class = {labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                             "support": int(sup[i])} for i in range(k)}
    present = sup > 0
    # most-confused pairs (off-diagonal)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(labels[i], labels[j], int(off[i, j])) for i, j in zip(*np.unravel_index(np.argsort(-off, None)[:10], off.shape)) if off[i, j] > 0]

    report = {"split": args.split, "n": int(len(y)), "top1": top1, "top5": top5,
              "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
              "per_class": per_class, "most_confused": pairs}
    (args.run / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2))
    plot_confusion(cm, labels, args.run / f"confusion_{args.split}.png",
                   f"{ckpt['model_name']} | {args.split} | top1={top1:.3f}")

    print(f"[{args.split}] n={len(y)}  top1={top1:.3f}  top5={top5:.3f}  macro-F1={report['macro_f1']:.3f}")
    worst = sorted(((v["f1"], n) for n, v in per_class.items() if v["support"]))[:5]
    print("lowest-F1 classes:", ", ".join(f"{n} ({s:.2f})" for s, n in worst))
    if pairs:
        print("most confused:", ", ".join(f"{a}->{b} x{c}" for a, b, c in pairs[:5]))
    print(f"wrote {args.run / f'eval_{args.split}.json'} and confusion_{args.split}.png")


if __name__ == "__main__":
    main()
