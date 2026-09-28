"""Extract MediaPipe Holistic landmarks from sign videos into a training .npz.

Supported layouts (--layout):
  flat     <root>/<label>/<video>                 (your own recordings)
  include  <root>/<Category>/<Word>/<video>       (INCLUDE / INCLUDE-50)

Output .npz keys:
  X        float32 [N, SEQ_LEN, FEAT_DIM]  normalised, resampled sequences
  y        int64   [N]                     index into `labels`
  labels   str     [K]                     class names (label order = model output order)
  signers  int64   [N]                     signer id parsed from filename, -1 if unknown
  paths    str     [N]                     source video paths

Example:
  python extract_landmarks.py --root D:/INCLUDE --layout include --out data/include50.npz
  python extract_landmarks.py --root recordings --layout flat --out data/own.npz --all-labels
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config as C
from features import (landmarks_to_array, normalize_sequence, part_present,
                      raw_frame_vector, resample_sequence, trim_inactive)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def normalise_label(name: str) -> str:
    """'48. Thank you' -> 'thank_you'."""
    s = re.sub(r"^\s*\d+\s*[.)_-]\s*", "", name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return C.LABEL_ALIASES.get(s.replace("_", ""), C.LABEL_ALIASES.get(s, s))


def find_videos(root: Path, layout: str) -> list[tuple[str, Path]]:
    depth = 1 if layout == "flat" else 2
    items = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS or not p.is_file():
            continue
        rel = p.relative_to(root).parts
        if len(rel) < depth + 1:
            continue
        # label = the folder directly containing the file at the expected depth
        items.append((normalise_label(rel[depth - 1]), p))
    return items


def parse_signer(path: Path, pattern: re.Pattern) -> int:
    m = pattern.search(path.stem)
    return int(m.group(1)) if m else -1


def process_video(path: Path, holistic, frame_stride: int, max_side: int) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % frame_stride == 0:
            h, w = bgr.shape[:2]
            if max(h, w) > max_side:            # speed: downscale large videos
                s = max_side / max(h, w)
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            frames.append(raw_frame_vector(
                landmarks_to_array(res.left_hand_landmarks),
                landmarks_to_array(res.right_hand_landmarks),
                landmarks_to_array(res.pose_landmarks, C.POSE_LANDMARK_IDS),
            ))
        i += 1
    cap.release()
    if not frames:
        return None
    seq = np.stack(frames)
    hand_frames = (part_present(seq, C.LH_SLICE) | part_present(seq, C.RH_SLICE)).sum()
    if hand_frames < C.MIN_HAND_FRAMES:
        return None
    seq = trim_inactive(seq)
    seq = normalize_sequence(seq)
    return resample_sequence(seq, C.SEQ_LEN)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--layout", choices=["flat", "include"], default="flat")
    ap.add_argument("--out", type=Path, default=C.DATA_DIR / "dataset.npz")
    ap.add_argument("--all-labels", action="store_true",
                    help="keep every folder label instead of filtering to config.VOCAB")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = no limit")
    ap.add_argument("--frame-stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-side", type=int, default=720, help="downscale frames larger than this")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--signer-regex", default=r"(?:^|[_\-. ])(?:signer|sgn|s)[_-]?(\d+)(?=$|[_\-. ])",
                    help="regex (group 1 = int) to parse signer id from the filename stem")
    args = ap.parse_args()

    import mediapipe as mp  # imported lazily so other scripts don't need it

    items = find_videos(args.root, args.layout)
    if not args.all_labels:
        vocab = set(C.VOCAB)
        skipped = sorted({l for l, _ in items if l not in vocab})
        items = [(l, p) for l, p in items if l in vocab]
        if skipped:
            print(f"[info] ignoring {len(skipped)} labels not in VOCAB (e.g. {skipped[:8]})")
    if args.max_per_class:
        counts: dict[str, int] = {}
        kept = []
        for l, p in items:
            if counts.get(l, 0) < args.max_per_class:
                kept.append((l, p))
                counts[l] = counts.get(l, 0) + 1
        items = kept
    if not items:
        sys.exit(f"No videos found under {args.root} with layout '{args.layout}'.")

    # Label order: VOCAB order for known words, then any extras alphabetically.
    present = {l for l, _ in items}
    labels = [w for w in C.VOCAB if w in present] + sorted(present - set(C.VOCAB))
    lab2idx = {l: i for i, l in enumerate(labels)}
    signer_re = re.compile(args.signer_regex, re.IGNORECASE)
    print(f"[info] {len(items)} videos, {len(labels)} classes")

    X, y, signers, paths, failed = [], [], [], [], []
    with mp.solutions.holistic.Holistic(static_image_mode=False,
                                        model_complexity=args.model_complexity,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as holistic:
        for label, path in tqdm(items, desc="extract"):
            try:
                seq = process_video(path, holistic, args.frame_stride, args.max_side)
            except Exception as e:  # corrupt video etc.
                print(f"[warn] {path}: {e}")
                seq = None
            if seq is None:
                failed.append(str(path))
                continue
            X.append(seq)
            y.append(lab2idx[label])
            signers.append(parse_signer(path, signer_re))
            paths.append(str(path))

    if not X:
        sys.exit("No usable sequences extracted.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=np.stack(X).astype(np.float32), y=np.array(y, dtype=np.int64),
                        labels=np.array(labels), signers=np.array(signers, dtype=np.int64),
                        paths=np.array(paths))
    print(f"[done] saved {len(X)} sequences -> {args.out}  (failed/no-hands: {len(failed)})")
    n_sig = len(set(signers) - {-1})
    print(f"[info] signer ids found: {n_sig}" + ("" if n_sig else " (use --signer-regex or name files *_signer03_*.mp4)"))


if __name__ == "__main__":
    main()
