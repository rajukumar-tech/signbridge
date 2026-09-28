"""Landmark-sequence augmentations (numpy, operate on normalised [T, F] arrays).

All transforms keep missing parts (all-zero blocks) at exactly zero, so the
"missing hand = zeros" convention survives augmentation.
"""
from __future__ import annotations

import numpy as np

import config as C
from features import resample_sequence


def _blocks(F: int) -> list[slice]:
    s = [C.LH_SLICE, C.RH_SLICE]
    if F >= C.POSE_SLICE.stop:
        s.append(C.POSE_SLICE)
    return s


def _presence(x: np.ndarray) -> np.ndarray:
    """(T, F) bool mask, True for coordinates belonging to a present block."""
    m = np.zeros_like(x, dtype=bool)
    for sl in _blocks(x.shape[1]):
        m[:, sl] = np.any(x[:, sl] != 0, axis=1, keepdims=True)
    return m


def mirror(x: np.ndarray) -> np.ndarray:
    """Left-right mirror: negate x, swap hand blocks, swap paired pose points.

    Note: for one-handed signs this turns a right-handed signer into a
    left-handed one, which is usually desirable (ISL signers use either hand).
    """
    out = x.copy().reshape(len(x), -1, 3)
    out[..., 0] *= -1
    out = out.reshape(len(x), -1)
    lh, rh = out[:, C.LH_SLICE].copy(), out[:, C.RH_SLICE].copy()
    out[:, C.LH_SLICE], out[:, C.RH_SLICE] = rh, lh
    if x.shape[1] >= C.POSE_SLICE.stop:
        pose = out[:, C.POSE_SLICE].reshape(len(x), -1, 3)
        for a, b in C.POSE_MIRROR_PAIRS:
            pose[:, [a, b]] = pose[:, [b, a]]
        out[:, C.POSE_SLICE] = pose.reshape(len(x), -1)
    return out


def rotate(x: np.ndarray, max_deg: float = 15.0, rng=np.random) -> np.ndarray:
    """In-plane (x, y) rotation about the origin (= shoulder midpoint after normalisation)."""
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(th), np.sin(th)
    mask = _presence(x)
    p = x.reshape(len(x), -1, 3).copy()
    xs, ys = p[..., 0].copy(), p[..., 1].copy()
    p[..., 0], p[..., 1] = c * xs - s * ys, s * xs + c * ys
    return np.where(mask, p.reshape(len(x), -1), 0).astype(np.float32)


def scale(x: np.ndarray, lo: float = 0.85, hi: float = 1.15, rng=np.random) -> np.ndarray:
    """Anisotropic scaling of x/y (and a milder z scale)."""
    sx, sy = rng.uniform(lo, hi), rng.uniform(lo, hi)
    sz = rng.uniform((1 + lo) / 2, (1 + hi) / 2)
    return (x.reshape(len(x), -1, 3) * np.array([sx, sy, sz], dtype=np.float32)).reshape(len(x), -1)


def translate(x: np.ndarray, max_shift: float = 0.1, rng=np.random) -> np.ndarray:
    shift = np.array([*rng.uniform(-max_shift, max_shift, 2), 0], dtype=np.float32)
    mask = _presence(x)
    return np.where(mask, (x.reshape(len(x), -1, 3) + shift).reshape(len(x), -1), 0).astype(np.float32)


def time_warp(x: np.ndarray, strength: float = 0.2, rng=np.random) -> np.ndarray:
    """Smooth monotonic re-timing (speed up / slow down parts of the sign), keeps length."""
    T = len(x)
    knots = 4
    # random positive speeds -> cumulative sum gives a monotonic time map
    speeds = np.clip(1 + rng.uniform(-strength, strength, knots), 0.2, None)
    cum = np.concatenate([[0], np.cumsum(speeds)])
    cum = cum / cum[-1] * (T - 1)
    grid = np.linspace(0, knots, T)
    positions = np.interp(grid, np.arange(knots + 1), cum)
    # also a small random crop of the start/end
    crop = rng.uniform(0, 0.1) * (T - 1)
    positions = crop / 2 + positions * (1 - crop / (T - 1))
    return resample_sequence(x, T, positions=positions)


def frame_dropout(x: np.ndarray, p_frame: float = 0.1, p_hand: float = 0.05, rng=np.random) -> np.ndarray:
    """Simulate tracking failures: drop whole frames (re-interpolated) and zero single hands."""
    T = len(x)
    keep = rng.random(T) > p_frame
    keep[0] = keep[-1] = True
    out = resample_sequence(x[keep], T) if keep.sum() < T else x.copy()
    for sl in (C.LH_SLICE, C.RH_SLICE):
        drop = rng.random(T) < p_hand
        out[drop, sl] = 0
    return out


def jitter(x: np.ndarray, sigma: float = 0.005, rng=np.random) -> np.ndarray:
    mask = _presence(x)
    return np.where(mask, x + rng.normal(0, sigma, x.shape).astype(np.float32), 0).astype(np.float32)


class Augmenter:
    """Randomly composes the transforms above. Call on a single [T, F] sample."""

    def __init__(self, p_mirror=0.5, p_rotate=0.5, p_scale=0.5, p_translate=0.3,
                 p_time_warp=0.5, p_dropout=0.3, p_jitter=0.5, seed: int | None = None):
        self.p = dict(mirror=p_mirror, rotate=p_rotate, scale=p_scale, translate=p_translate,
                      time_warp=p_time_warp, dropout=p_dropout, jitter=p_jitter)
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r, p = self.rng, self.p
        if r.random() < p["mirror"]:
            x = mirror(x)
        if r.random() < p["time_warp"]:
            x = time_warp(x, rng=r)
        if r.random() < p["rotate"]:
            x = rotate(x, rng=r)
        if r.random() < p["scale"]:
            x = scale(x, rng=r)
        if r.random() < p["translate"]:
            x = translate(x, rng=r)
        if r.random() < p["dropout"]:
            x = frame_dropout(x, rng=r)
        if r.random() < p["jitter"]:
            x = jitter(x, rng=r)
        return x.astype(np.float32)
