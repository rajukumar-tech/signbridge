"""Generate a small synthetic landmark dataset with the real .npz format.

It lets you smoke-test train -> evaluate -> export without any videos. The
data is NOT sign language: each class is a random but consistent pair of
hand trajectories + handshapes, with per-"signer" variation, noise, speed
changes and dropped hands. A model should reach high accuracy on it quickly.

  python make_synthetic.py --out data/synthetic.npz --classes 50 --per-class 24 --signers 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import config as C
from features import resample_sequence


def base_hand(curl: np.ndarray, size: float) -> np.ndarray:
    """21x3 crude hand: wrist + 5 fingers x 4 joints, fingers bent by `curl` (5,) in [0, 1]."""
    pts = [np.zeros(3)]
    angles = np.deg2rad([-50, -20, 0, 20, 40])           # finger spread, thumb first
    for f in range(5):
        d = np.array([np.sin(angles[f]), -np.cos(angles[f]), 0.0])
        p = d * 0.35 * size                               # knuckle
        pts.append(p.copy())
        for _ in range(3):
            d = d * (1 - curl[f] * 0.6) + np.array([0, 0.6 * curl[f], -0.3 * curl[f]])
            d /= np.linalg.norm(d) + 1e-8
            p = p + d * 0.2 * size
            pts.append(p.copy())
    return np.array(pts, dtype=np.float32)


def class_template(rng):
    """Random but fixed motion recipe for one class."""
    two_handed = rng.random() < 0.5
    hands = {}
    for side, sign in (("right", -1.0), ("left", 1.0)):       # right hand appears on image-left
        if side == "left" and not two_handed:
            continue
        hands[side] = dict(
            start=np.array([sign * rng.uniform(0.1, 0.6), rng.uniform(-0.2, 1.2), 0]),
            end=np.array([sign * rng.uniform(0.0, 0.7), rng.uniform(-0.4, 1.2), 0]),
            amp=rng.uniform(0, 0.25, 2), freq=rng.uniform(0.5, 3.0), phase=rng.uniform(0, 2 * np.pi),
            curl_a=rng.random(5), curl_b=rng.random(5),
        )
    return hands


def sample(template, signer, rng, T=45):
    """One raw-length (T) normalised sequence for a class template + signer style."""
    s_scale, s_off = signer
    t = np.linspace(0, 1, T)
    frames = np.zeros((T, C.FEAT_DIM), dtype=np.float32)
    if C.USE_POSE:
        pose = np.array([[0, -0.8, 0], [0.5, 0, 0], [-0.5, 0, 0], [0.6, 0.8, 0],
                         [-0.6, 0.8, 0], [0.4, 1.4, 0], [-0.4, 1.4, 0]], dtype=np.float32)
        frames[:, C.POSE_SLICE] = pose.reshape(-1)
    for side, h in template.items():
        path = h["start"] + (h["end"] - h["start"]) * t[:, None]
        wob = np.sin(2 * np.pi * h["freq"] * t + h["phase"])[:, None]
        path[:, :2] += h["amp"] * wob
        path = path * s_scale + s_off
        sl = C.LH_SLICE if side == "left" else C.RH_SLICE
        for i in range(T):
            curl = h["curl_a"] + (h["curl_b"] - h["curl_a"]) * t[i]
            hand = base_hand(curl, 0.5 * s_scale) + path[i]
            frames[i, sl] = hand.reshape(-1)
        # wrist z / pose z are flattish in MediaPipe; keep z small
    frames[:, :2 * C.HAND_DIM] += rng.normal(0, 0.01, (T, 2 * C.HAND_DIM)) * (frames[:, :2 * C.HAND_DIM] != 0)
    # random missed detections
    for sl in (C.LH_SLICE, C.RH_SLICE):
        frames[rng.random(T) < 0.05, sl] = 0
    # random speed: pick a sub-range and resample to SEQ_LEN
    a = rng.uniform(0, 0.1) * (T - 1)
    b = (T - 1) - rng.uniform(0, 0.1) * (T - 1)
    pos = a + (b - a) * np.linspace(0, 1, C.SEQ_LEN) ** rng.uniform(0.8, 1.25)
    return resample_sequence(frames, C.SEQ_LEN, positions=pos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=C.DATA_DIR / "synthetic.npz")
    ap.add_argument("--classes", type=int, default=len(C.VOCAB))
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--signers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = (C.VOCAB + [f"extra_{i}" for i in range(max(0, args.classes - len(C.VOCAB)))])[: args.classes]
    templates = [class_template(rng) for _ in labels]
    signer_styles = [(rng.uniform(0.85, 1.15), np.array([*rng.uniform(-0.1, 0.1, 2), 0])) for _ in range(args.signers)]

    X, y, signers, paths = [], [], [], []
    for c, tpl in enumerate(templates):
        for k in range(args.per_class):
            s = k % args.signers
            X.append(sample(tpl, signer_styles[s], rng, T=int(rng.integers(30, 70))))
            y.append(c)
            signers.append(s)
            paths.append(f"synthetic/{labels[c]}/{labels[c]}_signer{s:02d}_take{k}.mp4")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    X = np.stack(X).astype(np.float32)
    np.savez_compressed(args.out, X=X, y=np.array(y, dtype=np.int64), labels=np.array(labels),
                        signers=np.array(signers, dtype=np.int64), paths=np.array(paths))
    print(f"[done] X={X.shape} classes={len(labels)} signers={args.signers} -> {args.out}")


if __name__ == "__main__":
    main()
