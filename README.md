# SignBridge

Real-time Indian Sign Language (ISL) translator in the browser. It turns signs into text and speech, and text into an animated signing hand.

```
signbridge/
  demo/       Working in-browser prototype (MediaPipe hand tracking + k-NN, no build step)
  training/   Python pipeline: landmark extraction, augmentation, LSTM/Transformer, ONNX export
  backend/    FastAPI + PostgreSQL: accounts, "teach a sign" contributions, practice scores
```

## Run the demo (for the presentation)

```bash
python -m http.server 5173 --directory signbridge/demo
```

Open http://localhost:5173 in Chrome or Edge. The camera works on `localhost` without HTTPS.

1. **Start camera.**
2. **Teach a sign** tab: type `hello`, press **Record 4s**, and repeat the sign a few times. Do the same for `thank_you` (and any others).
3. Sign in front of the camera. Captions appear and are spoken aloud once a sign is held steadily.
4. **Text → Sign** tab: type `hello thank_you` and press Play to see the recorded signs replayed as an animated hand.
5. **Practice** tab: the app names a word; sign it within 5 seconds.

**Accounts (optional):** the demo works without logging in. To try **Sign up** / **Log in** (top right), also start the backend on port 8000. See [backend/README.md](backend/README.md). The pages are `demo/signup.html` and `demo/login.html`, and they call the API at `http://localhost:8000` (set in `demo/auth.js`).

Demo tips: use good front lighting and keep both hands in frame. Teach signs that look clearly different. Record each word 2–3 times, from slightly different positions. **Export dataset** saves what you taught, so you can re-import it on the presentation laptop.

The demo uses a per-user k-NN classifier so it can learn from a few seconds of data. The full project replaces it with the LSTM/Transformer from `training/`, exported to ONNX and run with onnxruntime-web.

## Full-project roadmap (4 weeks)

| Week | Goal |
|---|---|
| 1 | INCLUDE-50 subset + own clips → landmark `.npz` (`training/extract_landmarks.py`) |
| 2 | Train and compare BiLSTM vs Transformer, signer-independent evaluation, export ONNX |
| 3 | Next.js frontend with onnxruntime-web, FastAPI backend, text → sign avatar |
| 4 | User testing with a deaf school or NGO, Google Meet caption layer (browser extension / virtual camera), polish |

Dataset: INCLUDE, https://zenodo.org/records/4010759 (263 ISL word signs, 4,287 videos).
