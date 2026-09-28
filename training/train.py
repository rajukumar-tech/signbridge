"""Train a BiLSTM or Transformer ISL classifier on an extracted .npz dataset.

  python train.py --data data/include50.npz --model transformer --out runs/tfm
  python train.py --data data/own.npz --model lstm --split-by-signer

Writes to --out:
  best.pt        best checkpoint (by val accuracy) incl. model config + labels
  splits.npz     train/val/test indices (reused by evaluate.py)
  metrics.json   per-epoch history, best epoch, final test metrics
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

import config as C
from augment import Augmenter
from model import build_model, count_params


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["labels"] = [str(s) for s in out["labels"]]
    if "signers" not in out:
        out["signers"] = np.full(len(out["y"]), -1, dtype=np.int64)
    return out


def _strat_split(idx, y, frac, seed):
    """Stratified split; falls back to random when some class is too small."""
    try:
        return train_test_split(idx, test_size=frac, stratify=y[idx], random_state=seed)
    except ValueError:
        return train_test_split(idx, test_size=frac, random_state=seed)


def make_splits(y, signers, val_frac, test_frac, by_signer, seed):
    idx = np.arange(len(y))
    if by_signer and (signers >= 0).all() and len(np.unique(signers)) >= 3:
        # Signer-independent: whole signers are held out for test and for val.
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trval, te = next(gss.split(idx, y, groups=signers))
        rel_val = val_frac / (1 - test_frac)
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr, va = next(gss.split(trval, y[trval], groups=signers[trval]))
        tr, va = trval[tr], trval[va]
        mode = "signer-independent"
    else:
        if by_signer:
            print("[warn] --split-by-signer requested but signer ids missing/too few; "
                  "falling back to stratified random split")
        trval, te = _strat_split(idx, y, test_frac, seed)
        tr, va = _strat_split(trval, y, val_frac / (1 - test_frac), seed)
        mode = "stratified"
    return np.sort(tr), np.sort(va), np.sort(te), mode


class SeqDataset(Dataset):
    def __init__(self, X, y, augment: Augmenter | None = None):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment is not None:
            x = self.augment(x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.y[i])


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
@torch.no_grad()
def run_eval(model, loader, device, criterion=None):
    model.eval()
    n, correct, top5, loss_sum = 0, 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * len(y)
        k = min(5, logits.shape[1])
        tk = logits.topk(k, dim=1).indices
        correct += (tk[:, 0] == y).sum().item()
        top5 += (tk == y[:, None]).any(1).sum().item()
        n += len(y)
    return {"loss": loss_sum / max(n, 1), "top1": correct / max(n, 1), "top5": top5 / max(n, 1)}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", choices=["lstm", "transformer"], default="transformer")
    ap.add_argument("--out", type=Path, default=None, help="run dir (default runs/<model>)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--label-smoothing", type=float, default=C.LABEL_SMOOTHING)
    ap.add_argument("--warmup-epochs", type=int, default=C.WARMUP_EPOCHS)
    ap.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--val-frac", type=float, default=C.VAL_FRACTION)
    ap.add_argument("--test-frac", type=float, default=C.TEST_FRACTION)
    ap.add_argument("--split-by-signer", action="store_true",
                    help="hold out whole signers (needs signer ids in the .npz)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    out = args.out or (C.RUNS_DIR / args.model)
    out.mkdir(parents=True, exist_ok=True)

    d = load_npz(args.data)
    X, y, labels = d["X"].astype(np.float32), d["y"].astype(np.int64), d["labels"]
    assert X.ndim == 3 and X.shape[1] == C.SEQ_LEN, f"expected [N, {C.SEQ_LEN}, F], got {X.shape}"
    feat_dim, num_classes = X.shape[2], len(labels)
    tr, va, te, split_mode = make_splits(y, d["signers"], args.val_frac, args.test_frac,
                                         args.split_by_signer, args.seed)
    np.savez(out / "splits.npz", train=tr, val=va, test=te)
    print(f"[data] {len(y)} samples, {num_classes} classes, feat_dim={feat_dim} | "
          f"{split_mode}: train={len(tr)} val={len(va)} test={len(te)}")

    aug = None if args.no_augment else Augmenter(seed=args.seed)

    def loader(idx, augmenter, shuffle):
        return DataLoader(SeqDataset(X[idx], y[idx], augmenter), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=args.workers)

    train_dl, val_dl, test_dl = loader(tr, aug, True), loader(va, None, False), loader(te, None, False)

    model_kwargs: dict = {}
    model = build_model(args.model, num_classes, feat_dim, **model_kwargs).to(args.device)
    n_params = count_params(model)
    print(f"[model] {args.model}: {n_params:,} params on {args.device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_dl), 1)
    total, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):  # linear warmup, then cosine decay to 1% of the base LR
        if step < warm:
            return (step + 1) / max(warm, 1)
        prog = (step - warm) / max(total - warm, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history, best_acc, best_epoch, bad = [], -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, yb in train_dl:
            x, yb = x.to(args.device), yb.to(args.device)
            loss = criterion(model(x), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(yb)
            n += len(yb)
        val = run_eval(model, val_dl, args.device, criterion)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(n, 1), "val_loss": val["loss"],
               "val_top1": val["top1"], "val_top5": val["top5"], "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 2)}
        history.append(rec)
        print(f"ep {epoch:3d} | train {rec['train_loss']:.3f} | val {val['loss']:.3f} "
              f"top1 {val['top1']:.3f} top5 {val['top5']:.3f} | lr {rec['lr']:.2e} | {rec['sec']}s")

        if val["top1"] > best_acc:
            best_acc, best_epoch, bad = val["top1"], epoch, 0
            torch.save({"model_name": args.model, "model_kwargs": model_kwargs,
                        "state_dict": model.state_dict(), "labels": labels,
                        "feat_dim": feat_dim, "seq_len": C.SEQ_LEN, "num_classes": num_classes,
                        "epoch": epoch, "val_top1": best_acc, "data": str(args.data)},
                       out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] no val improvement for {args.patience} epochs")
                break

    ckpt = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test = run_eval(model, test_dl, args.device, criterion)
    metrics = {"model": args.model, "params": n_params, "split": split_mode,
               "n_train": len(tr), "n_val": len(va), "n_test": len(te), "num_classes": num_classes,
               "best_epoch": best_epoch, "best_val_top1": best_acc,
               "test_top1": test["top1"], "test_top5": test["top5"],
               "args": {k: str(v) for k, v in vars(args).items()}, "history": history}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] best val top1 {best_acc:.3f} @ epoch {best_epoch} | "
          f"test top1 {test['top1']:.3f} top5 {test['top5']:.3f} -> {out}")


if __name__ == "__main__":
    main()
