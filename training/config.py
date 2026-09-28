"""Shared configuration for the SignBridge ISL training pipeline.

Everything that must match the browser frontend (feature layout, sequence
length, normalisation constants, label order) lives here.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"            # extracted .npz datasets
RUNS_DIR = ROOT_DIR / "runs"            # checkpoints, metrics, plots
EXPORT_DIR = ROOT_DIR / "export"        # model.onnx + labels.json for the web app

# --------------------------------------------------------------------------
# Vocabulary (starter set of 50 common ISL words). Label index = position.
# --------------------------------------------------------------------------
VOCAB = [
    "hello", "thank_you", "help", "water", "doctor", "yes", "no", "please",
    "sorry", "good", "bad", "name", "what", "where", "how", "eat", "drink",
    "home", "school", "family", "mother", "father", "friend", "happy", "sad",
    "pain", "hospital", "emergency", "food", "bathroom", "today", "tomorrow",
    "time", "money", "work", "teacher", "student", "book", "deaf", "hearing",
    "sign", "learn", "understand", "again", "slow", "fast", "come", "go",
    "stop", "love",
]

# Folder-name aliases -> canonical vocab label (INCLUDE uses e.g. "48. Hello",
# "Thank you"; normalise_label() lower-cases, strips numbering and spaces).
LABEL_ALIASES = {
    "thankyou": "thank_you",
    "thanks": "thank_you",
    "toilet": "bathroom",
    "washroom": "bathroom",
    "mom": "mother",
    "dad": "father",
}

# --------------------------------------------------------------------------
# Feature layout (per frame)
#   [ left hand 21x3 | right hand 21x3 | pose 7x3 ]  = 63 + 63 + 21 = 147
# Missing hand/pose -> zeros.
# --------------------------------------------------------------------------
NUM_HAND_LANDMARKS = 21
COORDS = 3
HAND_DIM = NUM_HAND_LANDMARKS * COORDS          # 63

# MediaPipe Pose indices kept: nose, L/R shoulder, L/R elbow, L/R wrist
POSE_LANDMARK_IDS = [0, 11, 12, 13, 14, 15, 16]
POSE_NAMES = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]
POSE_DIM = len(POSE_LANDMARK_IDS) * COORDS      # 21
# Index pairs (within the 7 kept pose points) that swap when mirroring
POSE_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6)]
L_SHOULDER, R_SHOULDER = 1, 2                   # positions within kept pose points

USE_POSE = True
FEAT_DIM = 2 * HAND_DIM + (POSE_DIM if USE_POSE else 0)   # 147 (or 126)

LH_SLICE = slice(0, HAND_DIM)
RH_SLICE = slice(HAND_DIM, 2 * HAND_DIM)
POSE_SLICE = slice(2 * HAND_DIM, 2 * HAND_DIM + POSE_DIM)

SEQ_LEN = 30                                    # frames per sample after resampling

# Minimum number of frames with at least one hand detected for a clip to be kept
MIN_HAND_FRAMES = 3

# --------------------------------------------------------------------------
# Training defaults (overridable from the CLI)
# --------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 15
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
ONNX_OPSET = 17
