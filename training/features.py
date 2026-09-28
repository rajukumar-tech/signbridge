"""Frame feature construction, normalisation and temporal resampling.

This module is pure numpy (no MediaPipe import) so it can be reused by
augmentation / synthetic data, and it is the reference the browser
frontend's JavaScript feature code must reproduce exactly.

Per-frame normalisation (stateless, easy to mirror in JS):
  * centre  = shoulder midpoint if pose is present, else the wrist of the
              first present hand (left, then right)
  * scale   = shoulder width if pose is present, else the larger x/y extent
              of the present hand(s)
  * every *present* landmark: (p - centre) / scale  (x, y and z)
  * missing hands / pose stay exactly zero
"""
from __future__ import annotations

import numpy as np

import config as C

EPS = 1e-6


def landmarks_to_array(landmark_list, ids=None) -> np.ndarray | None:
    """MediaPipe NormalizedLandmarkList -> (K, 3) float32, or None if absent."""
    if landmark_list is None:
        return None
    lms = landmark_list.landmark
    if ids is not None:
        lms = [lms[i] for i in ids]
    return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)


def raw_frame_vector(left: np.ndarray | None, right: np.ndarray | None,
                     pose: np.ndarray | None) -> np.ndarray:
    """Stack raw (un-normalised) landmarks into a FEAT_DIM vector with zeros for missing parts."""
    v = np.zeros(C.FEAT_DIM, dtype=np.float32)
    if left is not None:
        v[C.LH_SLICE] = left.reshape(-1)
    if right is not None:
        v[C.RH_SLICE] = right.reshape(-1)
    if C.USE_POSE and pose is not None:
        v[C.POSE_SLICE] = pose.reshape(-1)
    return v


def part_present(frames: np.ndarray, sl: slice) -> np.ndarray:
    """Boolean mask (T,) — True where the block `sl` has any non-zero value."""
    return np.any(frames[..., sl] != 0, axis=-1)


def normalize_frame(v: np.ndarray) -> np.ndarray:
    """Normalise one raw frame vector (see module docstring). Returns a copy."""
    out = v.copy()
    lh = v[C.LH_SLICE].reshape(-1, 3)
    rh = v[C.RH_SLICE].reshape(-1, 3)
    has_l, has_r = np.any(lh != 0), np.any(rh != 0)
    has_pose = False
    if C.USE_POSE:
        pose = v[C.POSE_SLICE].reshape(-1, 3)
        has_pose = np.any(pose != 0)

    if not (has_l or has_r or has_pose):
        return out  # nothing detected: all zeros

    if has_pose:
        ls, rs = pose[C.L_SHOULDER], pose[C.R_SHOULDER]
        centre = (ls + rs) / 2.0
        scale = float(np.linalg.norm((ls - rs)[:2]))
    else:
        hand = lh if has_l else rh
        centre = hand[0].copy()                      # wrist
        pts = np.concatenate([h for h, ok in ((lh, has_l), (rh, has_r)) if ok])
        scale = float(np.max(pts[:, :2].max(0) - pts[:, :2].min(0)))
    scale = max(scale, EPS)

    blocks = [(C.LH_SLICE, has_l), (C.RH_SLICE, has_r)]
    if C.USE_POSE:
        blocks.append((C.POSE_SLICE, has_pose))
    for sl, ok in blocks:
        if ok:
            out[sl] = ((v[sl].reshape(-1, 3) - centre) / scale).reshape(-1)
    return out


def normalize_sequence(frames: np.ndarray) -> np.ndarray:
    return np.stack([normalize_frame(f) for f in frames]).astype(np.float32)


def _block_slices() -> list[slice]:
    s = [C.LH_SLICE, C.RH_SLICE]
    if C.USE_POSE:
        s.append(C.POSE_SLICE)
    return s


def resample_sequence(frames: np.ndarray, target_len: int = C.SEQ_LEN,
                      positions: np.ndarray | None = None) -> np.ndarray:
    """Resample (T, F) -> (target_len, F) by linear interpolation in time.

    Missing parts are handled per block: when both neighbouring source frames
    contain the block it is interpolated, otherwise the nearest frame's value
    (possibly zeros) is used, so we never blend real coordinates with zeros.
    `positions` optionally gives fractional source indices (used by time-warp).
    """
    T = len(frames)
    if T == 0:
        return np.zeros((target_len, frames.shape[-1]), dtype=np.float32)
    if T == 1:
        return np.repeat(frames, target_len, axis=0).astype(np.float32)
    if positions is None:
        positions = np.linspace(0, T - 1, target_len)
    positions = np.clip(positions, 0, T - 1)
    i0 = np.floor(positions).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (positions - i0)[:, None]
    nearest = np.where(w[:, 0] < 0.5, i0, i1)

    out = np.empty((target_len, frames.shape[-1]), dtype=np.float32)
    for sl in _block_slices():
        blk = frames[:, sl]
        present = np.any(blk != 0, axis=1)
        lerp = (1 - w) * blk[i0] + w * blk[i1]
        both = (present[i0] & present[i1])[:, None]
        out[:, sl] = np.where(both, lerp, blk[nearest])
    return out


def trim_inactive(frames: np.ndarray, pad: int = 2) -> np.ndarray:
    """Drop leading/trailing frames where no hand is visible (keeps `pad` frames of context)."""
    hands = part_present(frames, C.LH_SLICE) | part_present(frames, C.RH_SLICE)
    idx = np.flatnonzero(hands)
    if len(idx) == 0:
        return frames
    a, b = max(idx[0] - pad, 0), min(idx[-1] + pad + 1, len(frames))
    return frames[a:b]
