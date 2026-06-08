# ESL-software-ml Project Documentation
## Real-Time Egyptian Sign Language Recognition

---

## 1. Project Overview and Goals

### What is this project?

**ESL-software-ml** is a graduation project that builds a system to recognize **Egyptian Sign Language (ESL)** from a webcam or video file and display results on screen in real time — **without cloud services** and without a human interpreter.

### Problem it addresses

Communication between Deaf/hard-of-hearing users and hearing communities often depends on interpreters or text. This project provides a technical assistant that reads signs from video and outputs:

- **Sign label** (e.g. HELLO, THANKS, …)
- **Confidence score** (0.0–1.0)
- **Emotion** (in full desktop mode via DeepFace)

### Project phases

| Phase | Content | Status |
|-------|---------|--------|
| **Phase 1** | Data collection, landmark extraction, training, models v1/v2/v3 | Complete (Colab notebook) |
| **Phase 2** | Desktop app `demo.py` + `predict_frame()` API | Complete |
| **Phase 3** | Evaluation `src/evaluate.py`, ablation table, confusion matrix | Complete |
| **Phase 4** | Optional web app `web/` (FastAPI + WebSocket) | Optional |

### Main deliverables

1. **Desktop application** (`demo.py`) — three worker threads + OpenCV display.
2. **Stable API** (`predict_frame`) — for integration by a software team.
3. **Keras models** in `artifacts/` — primary runtime model: `model_v2.keras`.
4. **Scientific evaluation** under `results/`.

---

## 2. System Architecture

### High-level flow (Full mode)

```
┌─────────────┐     frame_queue      ┌──────────────────┐     result_queue     ┌─────────────────┐
│  Thread 1   │ ──────────────────► │    Thread 2      │ ──────────────────► │  Main Thread    │
│  Capture    │   (maxsize=2)       │    Inference     │   (maxsize=2)       │  Display (UI)   │
│  cv2.read   │                     │  predict_frame() │                     │  cv2.imshow    │
└─────────────┘                     └────────┬─────────┘                     └────────┬────────┘
                                             │                                        │
                                             │ MediaPipe Holistic                   │ latest_frame_ref
                                             │ Keras MLP (model_v2)                 │
                                             ▼                                        ▼
                                    ┌──────────────────┐                     ┌─────────────────┐
                                    │  _cached_emotion │ ◄── Thread 3 ──────│  Emotion Thread │
                                    │  (read by T2)    │     DeepFace       │  update_emotion │
                                    └──────────────────┘     every 5 polls   └─────────────────┘
```

### System layers

| Layer | Technology | Role |
|-------|------------|------|
| **Input** | OpenCV `VideoCapture` | Capture BGR frames from camera or file |
| **Landmarks** | MediaPipe `HolisticLandmarker` (Tasks API) | 21 joints/hand × 2 + 10 face points = 156 raw values |
| **Preprocessing** | Normalize + emotion one-hot | 163-dim model input |
| **Motion gate** | Activation gate (hand velocity) | Block prediction when hands are still |
| **Classifier** | Keras MLP (`model_v2.keras`) | Softmax → sign class |
| **Temporal smoothing** | Sliding window (5 frames) | Commit label after ≥3 votes and confidence ≥ 0.65 |
| **Emotion** | DeepFace (separate thread) | Update `_cached_emotion` asynchronously |
| **Display** | OpenCV overlay | Status, confidence, emotion, FPS, hand skeleton |

### Data path: frame → decision

1. Full-size BGR frame → resized to **320×240** inside `predict_frame`.
2. `HolisticLandmarker.detect_for_video()` → 156-dim vector.
3. No hands detected → `"No hand detected"` or `"__no_hands__"` (raw mode).
4. Hands still (velocity < 0.02) → `"Ready"`.
5. Hands moving → normalize → concat emotion → `model.predict` → `"Detecting"` or committed sign after voting.

### Runtime modes

| Mode | Command | Architecture |
|------|---------|--------------|
| **Full** | `python demo.py` | 3 threads + DeepFace + smoothing inside `inference.py` |
| **Sprint** | `python demo.py --sprint` | Single loop, fixed `neutral` emotion, smoothing in `demo.py` |

### Web app (`web/`)

One **async WebSocket handler** per connection: browser sends JPEG → server decodes → `predict_frame(frame, cached_emotion="neutral")` — **no DeepFace** and no separate emotion threads.

---

## 3. File Reference

### `src/` package

#### `src/paths.py`

**Purpose:** Resolve artifact paths consistently.

- Checks `artifacts/` then `output/artifacts/` (Colab/Cursor sprint layout).
- Returns the directory containing `model_v2.keras` or `label2idx.json`.
- Helpers: `artifacts_dir()`, `model_path()`, `label2idx_path()`, `results_dir()`.

**Why it matters:** Prevents breakage when models are copied to `output/artifacts/`.

---

#### `src/inference.py`

**Purpose:** Core of real-time inference.

| Function / constant | Description |
|-------------------|-------------|
| `load_model()` | Load Keras + `label2idx.json` + HolisticLandmarker; assert `input_shape[-1] == 163` |
| `predict_frame()` | Main contract: returns `(label, confidence, emotion)` |
| `update_emotion_async()` | DeepFace from Thread 3; writes `_cached_emotion` |
| `_resolve_emotion()` | Read cache or use passed `cached_emotion` (sprint/web) |
| `_extract_landmarks()` | MediaPipe → 156-dim vector |
| `_normalize()` | Wrist/face normalization — **must match Phase 1 Cell 9** |
| `_activation_gate()` | Hand velocity only (first 126 values) |
| `get_last_mp_results()` | Latest Holistic result for drawing without re-running |
| `reset_window()` | Clear voting window and motion gate |
| `set_frame_skip(n)` | Tune `FRAME_SKIP` for speed |

**Key constants:** `FACE_IDX`, `VELOCITY_THRESHOLD`, `WINDOW_SIZE`, `MIN_VOTES`, `MIN_CONFIDENCE`, `DEEPFACE_INTERVAL`, `FRAME_SKIP`.

---

#### `src/augmentation.py`

**Purpose:** Training-time data augmentation (Phase 1) — not used directly in `demo.py`.

| Function | Effect |
|----------|--------|
| `aug_hflip` | Horizontal flip + swap left/right hand blocks |
| `aug_time_jitter` | Speed up/slow sequence (0.8× / 1.2×) |
| `aug_gaussian_noise` | Gaussian noise on coordinates |
| `aug_landmark_dropout` | Random hand joint dropout (occlusion simulation) |
| `aug_rotation_2d` | Small rotation around wrist (optional) |
| `augment_sequence()` | All strategies on one sequence |
| `augment_dataset()` | Augment full `(N, T, D)` dataset |
| `lstm_to_mlp_features()` | Mean of 5 center frames → static MLP feature |
| `add_neutral_emotion()` | Append neutral one-hot `[0,0,0,0,1,0,0]` for training |

---

#### `src/evaluate.py`

**Purpose:** Phase 3 evaluation — model comparison and viva figures.

| Output | File |
|--------|------|
| Ablation table | `results/ablation_table.csv` |
| Confusion matrix | `results/confusion_matrix.png` |
| Per-class report | `results/classification_report.txt` |
| Failure analysis | `results/failure_analysis.png` |
| Memory log (optional) | `results/memory_log.csv` |

**Run:** `python -m src.evaluate`

---

#### `src/landmark_gate.py`

**Purpose:** Motion gate for **Sprint mode** only — uses `get_raw_landmarks()` without duplicating MediaPipe logic.

- `gate_from_frame(frame)` → `(is_active, raw_landmarks)`.
- Uses the same `VELOCITY_THRESHOLD` as `inference.py`.

---

### `demo.py`

**Purpose:** Desktop application entry point.

| Component | Description |
|-----------|-------------|
| `capture_thread` | Thread 1: read frames, `maxsize=2` queue, drop old frames |
| `inference_thread` | Thread 2: `predict_frame` + `get_last_mp_results` |
| `emotion_thread` | Thread 3: `update_emotion_async` every `DEEPFACE_INTERVAL` poll cycles |
| `run()` | Full mode: load model, start threads, display loop |
| `run_sprint()` | Single loop + `landmark_gate` + `predict_frame(..., raw=True)` |
| `draw_status_bar` | UI: sign, confidence, emotion, FPS |

**Keyboard:** Q quit, L landmarks, D debug, R reset voting window.

---

### `web/` folder

#### `web/server.py`

- **FastAPI** + **WebSocket** at `/ws`.
- On startup: `load_model()` once.
- Receives base64 JPEG → `cv2.imdecode` → `predict_frame(frame, cached_emotion="neutral")`.
- Responds with JSON: `label`, `confidence`, `emotion`, `latency_ms`, `fps_hint`.
- `/health` reports whether the model loaded.

#### `web/index.html`

- Simple UI: `getUserMedia` → 320×240 canvas → send JPEG every **100 ms** over WebSocket.
- Shows status, confidence, emotion (always neutral from server).

#### `web/manifest.json`

- PWA metadata only (no service worker).

**Run:**
```bash
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
```

---

### `artifacts/README.md` and `artifacts/` contents

| File | Description |
|------|-------------|
| `model_v1.keras` | Baseline MLP — no training augmentation |
| `model_v2.keras` | **Default runtime model** — augmented MLP + emotion slot |
| `model_v3.keras` | Bi-LSTM + augmentation + emotion — input `(30, 163)` |
| `label2idx.json` | Sign name → class index |
| `holistic_landmarker.task` | MediaPipe model (loaded in `load_model`) |
| `X_mlp_train.npy`, `X_mlp_val.npy` | MLP evaluation arrays (163 dims) |
| `X_lstm_train.npy`, `X_lstm_val.npy` | LSTM arrays `(samples, 30, 163)` |
| `y_train.npy`, `y_val.npy` | Integer labels |

---

## 4. Feature Vector (163 dimensions)

### Full structure

```
[ 156 normalized landmark values ] + [ 7 emotion one-hot values ] = 163
```

### Part 1: 156 landmarks (raw from MediaPipe before normalization)

| Segment | Index | Size | Content |
|---------|-------|------|---------|
| Left hand | 0 – 62 | 63 | 21 joints × (x, y, z) |
| Right hand | 63 – 125 | 63 | 21 joints × (x, y, z) |
| Face | 126 – 155 | 30 | 10 joints × (x, y, z) |

**Face indices used** (`FACE_IDX` — order must not change):

```
[0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
```

If a hand or face is not detected, that block is zero-filled.

### Normalization (`_normalize`)

| Part | Origin | Scale |
|------|--------|-------|
| Each hand | Wrist (joint 0) | Max distance from wrist |
| Face | Centroid of 10 points | Max distance from centroid |

**Warning:** Any mismatch with `normalize_frame()` in Phase 1 silently hurts accuracy.

### Part 2: 7 emotion dimensions (one-hot)

Class order:

```
angry, disgust, fear, happy, neutral, sad, surprise
```

Example `happy` → `[0, 0, 0, 1, 0, 0, 0]`  
Example `neutral` → `[0, 0, 0, 0, 1, 0, 0]`

- **Training (Phase 1):** Usually fixed `neutral` via `add_neutral_emotion()`.
- **Full runtime:** DeepFace updates label → dynamic one-hot.
- **Sprint / Web:** `cached_emotion="neutral"` always.

### Why 163 and not 156?

Models v2/v3 were designed to take **pose + emotional context** in one vector. `load_model()` rejects models whose last input dimension ≠ 163.

---

## 5. Emotion Detection

### Approach: DeepFace on a separate thread

Emotion is **not** inferred from MediaPipe landmarks in the current codebase; **DeepFace** analyzes full BGR frames.

### Flow

```
Thread 3 (emotion_thread)
    │
    ├─ Poll every 0.05 s
    ├─ Every DEEPFACE_INTERVAL (=5) poll cycles:
    │       update_emotion_async(latest_frame)
    │
    └─ Inside inference.py:
            DeepFace.analyze(frame, actions=["emotion"],
                             enforce_detection=False, silent=True)
            → dominant_emotion
            → _cached_emotion (protected by threading.Lock)

Thread 2 (predict_frame):
    emotion_str = _resolve_emotion(cached_emotion)
    features = concat(normalized_landmarks, one_hot(emotion_str))
```

### Technical details

| Aspect | Value / behavior |
|--------|------------------|
| Lazy import | `from deepface import DeepFace` on first call |
| Import failure | Function returns silently; emotion stays `neutral` |
| Frame failure | Exception swallowed; **last successful emotion kept** |
| `enforce_detection=False` | Analyze even with weak face detection |
| Sign/emotion conflict | `EMOTION_CONFLICTS` scales confidence (e.g. happy sign + angry face → ×0.75) |

### Frame source for DeepFace

`demo.py` sets `latest_frame_ref[0]` from the **display loop** after a result arrives from Thread 2 — the displayed frame, not necessarily the newest camera frame.

### Mode comparison

| Mode | Emotion source |
|------|----------------|
| Full (`demo.py`) | DeepFace → `_cached_emotion` |
| Sprint | Fixed `"neutral"` |
| Web | `cached_emotion="neutral"` on every request |

### Requirements

- `deepface` and `tf-keras` (with TensorFlow 2.16+) in `requirements.txt`.
- First run may download weights (~500 MB) to `~/.deepface`.

---

## 6. Threading

### Why multiple threads?

| Task | Cost | Thread |
|------|------|--------|
| Camera read | Low | T1 — not blocked by inference |
| MediaPipe + Keras | High | T2 |
| DeepFace | Very high | T3 — keeps UI responsive |

### Thread 1 — Capture (`T1-Capture`)

- `cv2.VideoCapture(source)`.
- `frame_queue` size 2: when full, **drop oldest** then enqueue newest → limits lag.

### Thread 2 — Inference (`T2-Inference`)

- Blocks on `frame_queue.get()`.
- Calls `predict_frame(frame)` → reads current `_cached_emotion` (updated asynchronously by T3).
- Pushes to `result_queue`: `(frame, label, conf, emotion, frame_count, mp_results)`.
- Same drop-old-frame policy when the queue is full.

### Thread 3 — Emotion (`T3-Emotion`)

- Does not consume `frame_queue` directly.
- Reads `latest_frame_ref[0]` (mutable one-element list for cross-thread sharing).
- `time.sleep(0.05)` → ~20 polls/s; every 5 polls → DeepFace.
- **Note:** Counter is **poll cycles**, not literal video frame indices.

### Main thread — Display

- Not a separate `threading.Thread` — the main `run()` loop.
- `result_queue.get()` → draw → `cv2.imshow` → `waitKey`.
- Updates `latest_frame_ref` for T3.

### Synchronization

| Resource | Mechanism |
|----------|-----------|
| `_cached_emotion` | `threading.Lock` on read/write |
| `_last_mp_results` | Written in T2; read-only in display (new object per `detect_for_video`) |
| Shutdown | `stop_event.set()` then `join(timeout=2)` per thread |

### Performance inside Thread 2

- **`FRAME_SKIP = 3`:** Reuse last prediction for 2 of every 3 frames — ~3× faster with small accuracy trade-off.

---

## 7. Models (model_v1, v2, v3)

### Comparison

| Model | Architecture | Training data | Input shape | Usage |
|-------|--------------|---------------|-------------|-------|
| **model_v1** | MLP | No augmentation (baseline) | `(N, 163)` | Baseline in ablation |
| **model_v2** | MLP | Augmented + neutral emotion at train time | `(N, 163)` | **Default** in `demo.py` and `load_model()` |
| **model_v3** | Bi-LSTM | Augmented + emotion | `(N, 30, 163)` | Temporal; evaluated on `X_lstm_val` |

### model_v1 — Baseline

- MLP on normalized landmarks + neutral one-hot.
- No `augmentation.py` pipeline in training.
- Reference “before improvements” in `evaluate.py`.

### model_v2 — Primary product model

- Same 163 dims, trained on augmented data (flip, jitter, noise, dropout).
- **Default path** in `_default_model_path()`.
- Matches `predict_frame()` (single frame → landmarks → MLP).

### model_v3 — LSTM

- Uses **30 frames** per sample.
- May capture motion better; heavier for live demo (not default in `demo.py`).
- Evaluated on `X_lstm_val.npy` in `evaluate.py`.

### Load-time validation

```python
expected_dim = 156 + 7  # 163
if model.input_shape[-1] != expected_dim:
    raise ValueError(...)
```

Prevents loading a 156-dim-only model or LSTM into the MLP inference path by mistake.

### Run comparison

```bash
python -m src.evaluate
```

Produces `results/ablation_table.csv` with Val Acc, Macro F1, Latency (ms).

---

## 8. How to Run the Project

### Requirements

- Python **3.10 or 3.11** (recommended; see `pyproject.toml`)
- Webcam or video file
- `artifacts/model_v2.keras`, `label2idx.json`, `holistic_landmarker.task`

### Environment setup (one-time — Windows)

```powershell
cd path\to\ESL-software-ml
.\setup_venv.ps1
```

Or manually:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**TensorFlow + DeepFace:** You may need `tf-keras`. If import fails with `jax` / `ml-dtypes` conflicts, uninstall `jax` and `jaxlib` from the venv.

### Quick check (no camera)

```powershell
python scripts/smoke_check.py
```

### Desktop — full mode (DeepFace emotions)

```powershell
python demo.py
python demo.py --source 0
python demo.py --source video.mp4
```

Or:

```powershell
.\run_demo.ps1
```

### Sprint mode (no DeepFace)

```powershell
python demo.py --sprint
```

Useful on locked-down networks or to avoid DeepFace downloads.

### Evaluation (Phase 3)

```powershell
python scripts/export_eval_arrays.py   # if landmarks exist but y_val is missing
python -m src.evaluate
```

### Web app

```powershell
.\run_web.ps1
# or:
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000`.

### Keyboard shortcuts in `demo.py`

| Key | Action |
|-----|--------|
| Q | Quit |
| L | Toggle hand skeleton |
| D | Debug mode |
| R | Reset voting window |

---

## 9. Important Design Decisions

### 1. Stable `predict_frame` contract

```python
(label: str, confidence: float, emotion: str)
```

The software team depends on this shape — changes require explicit coordination.

### 2. Exact match with Phase 1

- `FACE_IDX` order.
- `_normalize()` identical to notebook Cell 9.
- Drift degrades accuracy without a clear error.

### 3. Video-level train/val split (Phase 1)

Split by **video**, not frame — avoids data leakage and inflated accuracy.

### 4. Motion gate on hands only

Velocity uses landmarks 0:126 only — face motion (blink, speech) does not trigger prediction.

### 5. Sliding-window voting

- Window size 5, ≥3 votes for same label, mean confidence ≥ 0.65.
- Reduces label flicker in full mode.
- Sprint implements similar smoothing in `demo.py` instead of inside `inference`.

### 6. DeepFace off the sign hot path

Emotion is **not** computed inside `predict_frame` from landmarks — T3 + cache keeps inference fast.

### 7. Small queues and frame dropping

`maxsize=2` — prioritize the latest frame over processing every historical frame.

### 8. Reuse MediaPipe results for drawing

`get_last_mp_results()` — Holistic runs **once** per inference, not again for overlay.

### 9. Resize before Holistic

320×240 — balance between MediaPipe speed and landmark quality.

### 10. Configurable `FRAME_SKIP`

`set_frame_skip(1)` for accurate evaluation; default `3` for live demo.

### 11. Fail fast on model input shape

Check in `load_model()` — clearer than a shape error deep in `predict`.

### 12. Neutral emotion at training time

`add_neutral_emotion()` — model learns an emotion slot; runtime can fill it from DeepFace.

### 13. Semantic sign/emotion conflicts

`EMOTION_CONFLICTS` — lower confidence when sign and face emotion disagree.

### 14. MediaPipe Tasks API (not legacy `mp.solutions`)

MediaPipe ≥ 0.10.30 removed old holistic — project uses `HolisticLandmarker` + `.task` file.

### 15. Flexible artifact paths

`src/paths.py` supports both `artifacts/` and `output/artifacts/`.

### 16. Web: CPU-only and simple

`cached_emotion="neutral"` — no threads or DeepFace on the server to reduce complexity and load.

---

## Appendix: Repository layout

```
ESL-software-ml/
├── demo.py                 # Desktop app
├── PROJECT_DOCS.md         # This file
├── requirements.txt
├── src/
│   ├── inference.py        # Inference + DeepFace cache
│   ├── augmentation.py     # Data augmentation
│   ├── evaluate.py         # Evaluation
│   ├── paths.py            # Artifact paths
│   └── landmark_gate.py    # Sprint motion gate
├── web/
│   ├── server.py
│   └── index.html
├── artifacts/              # Models and arrays
└── results/                # Evaluation outputs
```

---

*Last updated: reflects DeepFace on Thread 3 and `model_v2.keras` as the default runtime model.*
