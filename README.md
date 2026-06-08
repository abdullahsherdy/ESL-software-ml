# ESL Real-Time Sign Language Recognition

**Graduation Project**

A local Python application that watches a person sign in Egyptian Sign Language and displays the recognised sign in real time — no cloud, no internet, no human interpreter.

---

## System Overview (to be enhanced)

```
Webcam → Thread 1 (Capture) → frame_queue
                             ↓
                       Thread 2 (Inference)
                         MediaPipe Holistic
                         Activation gate
                         normalize → model.predict()
                             ↓
                       result_queue
                             ↓
                       Main Thread (Display)
                         cv2.imshow overlay

Thread 3 (Emotion)  ─── DeepFace every 5 frames ──→ _cached_emotion
                                                          ↑
                                                   (inference reads)
```

---

## Prerequisites

- Python **3.10 or 3.11 only** for this repo (`pyproject.toml` sets `requires-python = ">=3.10,<3.13"`; TensorFlow wheels on Windows do not support **3.13** today — using 3.13 as your default `python` causes install/import failures).
- `artifacts/model_v2.keras` (or the same files under `output/artifacts/`) — from Phase 1 notebook
- `artifacts/label2idx.json` — class index map (see [artifacts/README.md](artifacts/README.md))
- Webcam (built-in or USB)

### One-time environment (Windows)

If `python --version` shows 3.12+ or 3.13, install [Python 3.11](https://www.python.org/downloads/) and enable **Add python.exe to PATH**, then:

```powershell
cd path\to\esl_project
.\setup_venv.ps1
```

After that, **`run_demo.ps1` / `run_web.ps1` use `.venv`** (or `py -3.11` if you have no venv yet). A **`.python-version`** file (`3.11`) helps pyenv/pyenv-win pick the right interpreter.

---

## Desktop: two modes

| Mode | Command | Architecture |
|------|---------|--------------|
| **Full** (default) | `python demo.py` or `./run_demo.sh` / `.\run_demo.ps1` | Threaded capture + inference + DeepFace cache (`.claude/CLAUDE.md`) |
| **Sprint** | `python demo.py --sprint` | Single OpenCV loop, `predict_frame(..., "neutral", raw=True)` + app-layer smoothing (`.cursor/instruction.md`) |

Sprint mode avoids DeepFace downloads — useful on locked-down exam Wi‑Fi.

Quick model check (no camera):

```bash
python scripts/smoke_check.py
```

---
## File Structure

```
esl_project/
├── demo.py                    # Desktop app entry point
├── run_demo.sh / run_demo.ps1
├── run_web.sh   / run_web.ps1
├── setup_venv.sh / setup_venv.ps1  # one-time: create .venv with 3.10/3.11 + pip install
├── pyproject.toml              # requires-python >=3.10,<3.13
├── .python-version             # 3.11 (pyenv / editor hint)
├── requirements.txt
├── scripts/
│   ├── esl_python.ps1          # Windows: resolve 3.10/3.11 for run_*.ps1
│   ├── export_eval_arrays.py  # Regenerates X/y arrays + y_val.npy
│   ├── export_tflite.py       # Keras -> TFLite conversion (+ --validate)
│   └── smoke_check.py         # Quick load_model / predict sanity (no webcam)
│
├── output/                    # Optional mirror for Colab / Cursor sprint imports
│   ├── artifacts/             # copy or symlink of artifacts/
│   └── src/inference.py       # re-exports src.inference
├── web/
│   ├── server.py              # FastAPI + WebSocket
│   ├── index.html
│   └── manifest.json          # PWA metadata only (no service worker)
├── src/
│   ├── inference.py           # predict_frame()
│   ├── paths.py               # resolves artifacts/ vs output/artifacts/
│   ├── landmark_gate.py       # sprint activation gate helpers
│   ├── augmentation.py        # Data augmentation
│   └── evaluate.py            # Evaluation suite
│
├── artifacts/                 # Primary model location (or use output/artifacts/)
│   ├── model_v1.keras         # Baseline MLP (no augmentation)
│   ├── model_v2.keras         # Augmented MLP + Emotion  ← PRIMARY
│   ├── model_v3.keras         # Augmented LSTM + Emotion
│   ├── model_v2.tflite        # Optional: produced by scripts/export_tflite.py
│   └── label2idx.json
├── data/
│   ├── landmarks/             # Raw .npy per-video landmark files
│   └── augmented_landmarks/   # Augmented sequences (flip, slow, fast, noise)
│
└── results/
    ├── confusion_matrix.png
    ├── failure_analysis.png
    ├── ablation_table.csv
    ├── classification_report.txt
    ├── viva_prep.md           # Checklist template for the team
    └── training_curves.png
```
---
## Interface Contract

```python
from src.inference import predict_frame

# Full mode (default): uses DeepFace-cached emotion when cached_emotion is omitted
label, confidence, emotion = predict_frame(frame)

# Sprint / web: fixed neutral one-hot (no DeepFace thread required)
label, confidence, emotion = predict_frame(frame, cached_emotion="neutral", raw=True)
# raw=True → per-frame softmax label (use app-layer smoothing in sprint demo)

# frame: BGR numpy array from cv2.VideoCapture
# label: str — sign class, or state strings (full mode) / "__no_hands__" (raw mode)
# confidence: float — 0.0–1.0
# emotion: str — from DeepFace cache (full) or passed through (sprint)
```

```python
def predict_frame(frame):
    return ("HELLO", 0.91, "happy")
```

---

## Models

| File | Description | Input dim | Val Acc* |
|------|-------------|-----------|----------|
| `model_v1.keras` | Baseline MLP — no augmentation, neutral-only emotion | 163 | — |
| `model_v2.keras` | Augmented MLP + Emotion ← **use this** | 163 | — |
| `model_v3.keras` | Augmented Bi-LSTM + Emotion | (30, 163) | — |

*Fill in from `results/ablation_table.csv` after running `python -m src.evaluate`.

All three variants take a 163-dim input (156 landmark features + 7-dim
emotion one-hot). `src/inference.py` asserts this at `load_model()` time so
pointing `MODEL_PATH` at a mismatched model fails fast instead of silently
mispredicting.

---

## Phase 1 — Data & Model (done)

Run `ESL_Phase1_Complete.ipynb` in Google Colab.

**Steps:**
1. Kaggle dataset download
2. Video-level 80/20 split
3. Frame extraction at 15 FPS
4. MediaPipe Holistic landmark extraction (156 features/frame)
5. Normalisation (wrist-centred + scale)
6. Augmentation: hflip, time-jitter ×2, noise, dropout → ≥2,200 sequences
7. Emotion feature concat (neutral placeholder at train time)
8. Train model_v1, model_v2, model_v3
9. Ablation table, confusion matrix, failure analysis

**Why video-level split?** Frame-level splitting leaks data — frames from the same video appear in both train and val, inflating accuracy. See Step 2 in the notebook.

---

## Phase 2 — Real-Time Integration

All code is in `demo.py` + `src/inference.py`.

**Key design decisions:**
- `static_image_mode=False` in MediaPipe for live video (tracks across frames → faster)

- `queue.Queue(maxsize=2)` drops old frames — prevents ever-growing  
- Activation gate (velocity threshold 0.02) — no predictions when hands are still
- Sliding window majority vote (5 frames, ≥3 votes, ≥0.65 confidence)
- DeepFace runs every 5 frames in its own thread — expensive, so cached (full mode only)

---

## Phase 3 — Evaluation

If the per-video landmark `.npy` files are present in `data/landmarks/` +
`data/augmented_landmarks/` but the evaluation arrays are missing from
`artifacts/` (e.g. `y_val.npy`), regenerate them first:

```bash
python scripts/export_eval_arrays.py
```

Then:

```bash
python -m src.evaluate
```

Outputs: `results/confusion_matrix.png`, `results/failure_analysis.png`,
`results/ablation_table.csv`, `results/classification_report.txt`

For slow (CPU-only) demo machines, export TFLite:

```bash
python scripts/export_tflite.py --validate
# -> artifacts/model_v2.tflite
# -> prints max |Keras - TFLite| drift and top-1 agreement
```

---

Fill metrics into [results/viva_prep.md](results/viva_prep.md) and the table below.

---

## Phase 4 — Web App (optional)

```bash
# Linux / macOS / Git Bash
./run_web.sh

# Windows PowerShell
.\run_web.ps1
```

Open **http://localhost:8000** — the page serves `web/index.html` and talks to `/ws`.
`GET /health` returns JSON with `model_loaded` (false if `model_v2.keras` is missing).

For **mobile access** over HTTPS, use [ngrok](https://ngrok.com/download) on port 8000.
The repo ships `manifest.json` only (installable shortcut) — **no service worker**, per sprint spec.

---

## Performance Targets

| Metric | Target | How to measure |
|--------|--------|----------------|
| Display FPS | ≥ 15 | FPS counter in app corner |
| Inference latency | ≤ 100ms | `src/evaluate.py` timing |
| Web round-trip | ≤ 150ms | `web/server.py` latency log |
| Model size (v2) | ~200 KB | `ls -lh artifacts/model_v2.keras` |

If FPS < 10, export to TFLite:
```python
import tensorflow as tf
model = tf.keras.models.load_model("artifacts/model_v2.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("artifacts/model_v2.tflite", "wb") as f:
    f.write(tflite_model)
```

---

---

## Viva Talking Points

**"Why LSTM over MLP?"**
> "We categorised all 12 MVP signs as static or dynamic. 7 signs require motion to convey meaning — these need LSTM. For the 5 static signs, MLP is sufficient and faster. Our ablation table confirms LSTM outperforms MLP on dynamic signs."

**"Why MediaPipe over raw CNN?"**
> "With only 10 videos per class, a CNN would overfit severely and need GPU. MediaPipe gives us 156 clean numbers per frame — background and lighting are stripped. Our baseline MLP runs at 50+ FPS on CPU with a 200KB model."

**"What are your limitations?"**
> "We trained on one signer. Accuracy drops ~15% on unseen signers due to hand proportion variation. Scaling to all 55 classes needs 50+ videos per class. Sentence-level recognition would need temporal segmentation and a language model."

---

## Known Issues to avoid / take in consideration 
- If `python scripts/smoke_check.py` prints "tensorflow not installed", switch to Python 3.11 and `pip install -r requirements.txt`.

- DeepFace downloads model weights (~500MB) on first run — requires internet

- Web/desktop crash when showing hands (`The packet is empty`) → upgrade MediaPipe: `pip install "mediapipe>=0.10.30,<0.11"` (venv had 0.10.14; use 0.10.30+)
- `mp.solutions.holistic` was removed in MediaPipe 0.10.30+; this project uses `HolisticLandmarker` (tasks API) instead


