# SignBridge — ISL model training

Trains a small sequence classifier on MediaPipe landmarks for SignBridge's browser-based Indian Sign Language translator, then exports it to ONNX for `onnxruntime-web`.

Pipeline: `extract_landmarks.py` (videos → `.npz` [N,30,147]) → `train.py` (`best.pt`) → `evaluate.py` (top-1/5, F1, confusion PNG) → `export_onnx.py` (`model.onnx` + `labels.json`). You can use `make_synthetic.py` in place of real videos to smoke-test the pipeline.

## Files
| File | Purpose |
|---|---|
| `config.py` | Vocabulary (50 words), feature layout, sequence length, paths, training defaults |
| `features.py` | Frame vector, normalisation, resampling. **The JS frontend must reproduce this exactly.** |
| `extract_landmarks.py` | Videos → MediaPipe Holistic → normalised 30-frame sequences → `.npz` |
| `augment.py` | Mirroring, rotation, scaling, translation, time-warp, frame/hand dropout, jitter |
| `model.py` | `BiLSTMClassifier` (~705k params), `TransformerClassifier` (~295k params) |
| `train.py` | AdamW, warmup + cosine LR, label smoothing, early stopping, optional signer-independent split |
| `evaluate.py` | Top-1/top-5, per-class P/R/F1, most-confused pairs, confusion-matrix PNG |
| `export_onnx.py` | ONNX export (`[1,30,147]` → softmax probs), onnxruntime check, `labels.json` |
| `make_synthetic.py` | Synthetic dataset with the real shapes |

## Feature format (must match the frontend)

Each frame is 147 floats: `[left hand 21×3 | right hand 21×3 | pose 7×3]`.

**Pose points.** MediaPipe Pose ids 0 and 11–16: nose, then left/right shoulder, elbow and wrist.

**Normalisation.** This is done per frame:
- Centre on the shoulder midpoint and scale by the shoulder width (x/y only).
- If there is no pose, centre on the wrist of the first present hand and scale by the largest x/y extent of the present hands.
- Every present landmark (x, y and z) becomes `(p − centre) / scale`.
- A missing part is all zeros and stays zero.

**Resampling.** Frames with no hands are trimmed from the start and end. The rest is linearly resampled to 30 frames. Each part is resampled separately, so real coordinates are never blended with zeros.

**Left/right hands.** Use Holistic's `leftHandLandmarks` / `rightHandLandmarks` exactly as in training. If you show a mirrored selfie preview, mirror only the display, never the frames you feed to MediaPipe.

`USE_POSE = False` in `config.py` gives 126 features (hands only). Re-extract after changing it.

## Step by step

### 0. Environment
```bash
python -m venv C:\venvs\signbridge
C:\venvs\signbridge\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
The `C:\venvs\signbridge` location matters. On Windows, `import torch` fails with `WinError 206 (filename or extension is too long)` when the venv sits under a very long path, such as this scratch workspace.

### 1. Smoke test (no data needed, about 1 minute on CPU)
```bash
python make_synthetic.py --out data/synthetic.npz
python train.py --data data/synthetic.npz --model transformer --epochs 2 --out runs/smoke
python evaluate.py --run runs/smoke
python export_onnx.py --run runs/smoke --out export
```

### 2. Get INCLUDE
1. Download it from Zenodo: https://zenodo.org/records/4010759. It comes as per-category zips. Read the licence on the record first.
2. Unzip everything to a single `INCLUDE/<Category>/<NN. Word>/<video>` tree.

The target subset is INCLUDE-50.

### 3. Record your own clips (recommended)
`config.VOCAB` only partly overlaps with INCLUDE-50, so you will need your own clips for some words. Webcam clips also reduce the domain gap.

- **Layout.** Save clips as `recordings/<label>/<label>_signer03_take07.mp4`. The `signerNN` part is what enables `--split-by-signer`.
- **Amount.** Aim for at least 5 signers × 5 takes per word.
- **Variety.** Vary lighting, background and distance, and include left-handed signers.
- **Clip content.** One sign per clip, 1–3 s long.

### 4. Extract landmarks
```bash
python extract_landmarks.py --root D:/INCLUDE --layout include --out data/include_vocab.npz          # keep VOCAB words only
python extract_landmarks.py --root D:/INCLUDE50 --layout include --all-labels --out data/include50.npz  # native labels
python extract_landmarks.py --root recordings --layout flat --out data/own.npz
```
Folder names are normalised to labels, so `48. Hello` becomes `hello`.

Useful flags:
- `--frame-stride` to skip frames
- `--max-side` to cap frame size
- `--model-complexity` for the Holistic model
- `--signer-regex` to change how signer ids are parsed

Clips with fewer than 3 hand frames are skipped.

### 5. Train
```bash
python train.py --data data/include50.npz --model transformer --out runs/tfm
python train.py --data data/include50.npz --model lstm --out runs/lstm
python train.py --data data/own.npz --model transformer --split-by-signer --out runs/own_si
```
- **Default split.** Stratified 70/15/15.
- **`--split-by-signer`.** Holds out whole signers. It needs signer ids and at least 3 signers; otherwise it warns and falls back to the stratified split.
- **Outputs.** `best.pt`, `splits.npz`, `metrics.json`.

### 6. Evaluate
```bash
python evaluate.py --run runs/tfm
python evaluate.py --run runs/tfm --split all --data data/own.npz
```
The first command scores the saved test split. The second checks the model on another dataset.

Each run writes `eval_<split>.json` and `confusion_<split>.png`.

### 7. Export for the browser
```bash
python export_onnx.py --run runs/tfm --out export
```
- **`model.onnx`.** Input `landmarks` float32 `[1,30,147]`. Output `probs` float32 `[1,num_classes]`, with softmax already applied.
- **`labels.json`.** Label order plus the feature and normalisation metadata.
- **Verification.** The script compares onnxruntime against PyTorch and fails if the max abs diff is ≥ 1e-4.
- **`--dynamic-batch`.** Adds a variable batch axis.

#### Loading with onnxruntime-web
```js
import * as ort from "onnxruntime-web";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/"; // or self-host the .wasm files
const session = await ort.InferenceSession.create("/models/model.onnx", { executionProviders: ["wasm"] }); // or "webgpu"
const meta = await (await fetch("/models/labels.json")).json();
// win: Float32Array(30*147) built with the SAME normalisation as features.py
const { probs } = await session.run({ landmarks: new ort.Tensor("float32", win, [1, meta.seq_len, meta.feat_dim]) });
const best = probs.data.indexOf(Math.max(...probs.data));
console.log(meta.labels[best], probs.data[best]);
```
In the frontend:
- Keep a rolling buffer of 1–2 s and resample it to 30 frames.
- Only run inference while hands are visible.
- Accept a prediction only if its probability is above about 0.6 and it stays the same across 2–3 windows.

## Expected results and honest notes

**Synthetic data only checks the plumbing.** The 2-epoch smoke test scores about 20% top-1, which is expected.

**INCLUDE-50 baselines.** Published baselines are in Sridhar et al., "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition" (ACM Multimedia 2020). Compare against its tables. Scores far above the paper's usually mean a signer or clip leaked across the splits.

**New users will score lower.** Signer-independent (new-person, webcam) accuracy is typically 10–30 points below random-split accuracy. That is the number that matters for SignBridge. Multi-signer webcam data is the best way to improve it.

**Limitations.**
- Hands plus 7 pose points cannot capture facial expression or mouthing.
- The model only recognises isolated signs.
- Consider adding a `no_sign` class to reduce false positives in live use.

This is an assistive prototype, not a certified interpreter. Validate it with Deaf ISL users, especially for critical words like *emergency*, *doctor* and *pain*.
