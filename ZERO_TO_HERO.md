# ESL Sign Language Recognition — Zero to Hero Guide
> Complete technical documentation: data → model → inference → deployment.
> Written for viva preparation, teammate onboarding, and future contributors.
> Verified against codebase: 2026-06-09

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Why These Technical Choices](#2-why-these-technical-choices)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Engineering — Landmarks to Numbers](#4-feature-engineering--landmarks-to-numbers)
5. [Data Augmentation](#5-data-augmentation)
6. [Model Architecture](#6-model-architecture)
7. [Training Process](#7-training-process)
8. [Evaluation Results](#8-evaluation-results)
9. [Inference Engine — predict_frame()](#9-inference-engine--predict_frame)
10. [Desktop Application](#10-desktop-application)
11. [Web Application](#11-web-application)
12. [Mobile Deployment — TFLite](#12-mobile-deployment--tflite)
13. [Cloud Deployment — Railway](#13-cloud-deployment--railway)
14. [Bug History and Fixes](#14-bug-history-and-fixes)
15. [Viva Q&A — Prepared Answers](#15-viva-qa--prepared-answers)

---

## 1. What This Project Does

### The Problem

Deaf and hard-of-hearing people communicate through sign language. When interacting with hearing people who don't know sign language, they typically need a human interpreter. This project builds a software interpreter that reads Egyptian Sign Language (ESL) from a camera in real time and displays the recognised word on screen — with no internet connection, no human interpreter, and no specialised hardware.

### The Output

Given a video stream from a webcam or phone camera, the system outputs:

```
("HELLO", 0.89, "neutral")
  │         │        └── detected emotion of the signer
  │         └─────────── model's confidence in this prediction (0–1)
  └───────────────────── the recognised Egyptian Sign Language word
```

### The Scope

| Capability | Current state |
|------------|--------------|
| Sign classes | 47 Egyptian signs (of 55 total) |
| Signers in training | 1 (primary limitation — see §15) |
| Inference hardware | CPU only — no GPU required |
| Model size | 973 KB (MLP v2) |
| Languages supported | Egyptian Sign Language (ESL) |

---

## 2. Why These Technical Choices

### Why not a CNN on raw video frames?

A Convolutional Neural Network (CNN) on raw images needs **thousands of examples per class** to generalise. We have approximately 10 videos per sign class. A CNN trained on this data would:

- Overfit to the specific signer's skin tone, background, and lighting
- Need a GPU to run at real-time speed
- Be 14+ MB in size (vs our 973 KB)

Instead, we use **MediaPipe** to extract the 3D coordinates of every hand joint and selected face points. This gives us 156 clean numbers per frame, stripped of all background and lighting variation. Our MLP then classifies these 156 numbers — a task where 10 videos per class is sufficient.

### Why MediaPipe?

MediaPipe Holistic is a production-grade computer vision library from Google. It runs on CPU, handles partial occlusions, and outputs normalised 3D landmark coordinates. Crucially it separates the perception problem (where are the hands?) from the classification problem (what sign is this?) — letting us focus a small model on the classification task only.

### Why an MLP and not a deeper network?

With 156 input features and 47 classes, a 3–4 layer MLP with ~163K parameters is already expressive enough to reach 91.74% accuracy. Deeper networks would overfit on our dataset size. The MLP also runs in ~5ms per inference on CPU — fast enough for real-time use even on slow hardware.

### Why an LSTM for some signs?

Static signs (where meaning comes only from hand shape) are well-served by an MLP that classifies a single frame. Dynamic signs (where meaning comes from the motion trajectory) need a model that sees a sequence of frames. Our LSTM (model v3) reads 30 consecutive frames — capturing temporal patterns like direction of movement, speed, and shape change.

### Why landmark normalisation?

Raw MediaPipe coordinates are in screen space — they change with how far the signer stands from the camera and where their hands are on screen. A signer with large hands at close range produces very different raw numbers than a signer with small hands at arm's length, even for the same sign. Normalisation (wrist as origin, scale by max joint distance) removes these confounders and lets the model focus on hand shape only.

---

## 3. Data Pipeline

### 3.1 Dataset Structure

```
data/
├── landmarks/
│   ├── HELLO_signer1_take1.npy      ← raw landmark array per video
│   ├── HELLO_signer1_take2.npy
│   ├── HELLO_signer1_take1_aug_flip.npy
│   ├── HELLO_signer1_take1_aug_slow.npy
│   └── ...
└── augmented_landmarks/
```

Each `.npy` file contains the landmark sequence for one video — shape `(T, 156)` where `T` is the number of frames extracted from that video.

### 3.2 What MediaPipe Extracts Per Frame

MediaPipe Holistic detects and tracks:
- 21 landmarks per hand (left and right)
- 478 face landmarks (we use 10 selected ones)
- 33 body pose landmarks (we do not use these)

Each landmark has three coordinates: x (width), y (height), z (depth, relative to hip for pose, relative to wrist for hands).

**What we keep:**

```
Per frame → 156 numbers:
  Indices   0– 62:  left  hand  (21 joints × 3 coords = 63)
  Indices  63–125:  right hand  (21 joints × 3 coords = 63)
  Indices 126–155:  face        (10 points  × 3 coords = 30)
```

**The 10 face landmarks we select (from MediaPipe's 478):**
```python
FACE_IDX = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
```
These are nose tip, between eyebrows, corners of mouth, chin, and eye corners — enough to capture broad facial expression without 468 redundant points.

**If a hand is not detected:** all 63 values for that hand are set to 0.0.
**If face is not detected:** all 30 values are set to 0.0.

### 3.3 The Data Leakage Bug (and how we fixed it)

**The wrong way** (frame-level split):
```
Video A frames: [f1, f2, f3, f4, f5, ...]
→ Train: [f1, f3, f5, ...]
→ Val:   [f2, f4, f6, ...]
```

This is wrong because frames from the same video share the signer's hand proportions, arm length, and camera angle. The model memorises the signer's identity, not the sign shape. Validation accuracy is artificially inflated.

**The correct way** (video-level split):
```
Videos for class HELLO: [take1, take2, take3, take4, take5]
→ Train: [take1, take2, take3, take4]  (80%)
→ Val:   [take5]                        (20%)
```

No frame from a validation video ever appears in training — not even augmented versions. The model must generalise to an unseen video to score a validation point.

### 3.4 Train/Val Split Numbers

| Split | MLP shape | LSTM shape |
|-------|-----------|------------|
| Train | (N_train, 163) | (N_train, 30, 163) |
| Val   | (121, 163) | (121, 30, 163) |

121 validation samples across 47 classes (approximately 2–3 val samples per class).

---

## 4. Feature Engineering — Landmarks to Numbers

### 4.1 Normalisation

Before feeding landmarks to the model, we normalise them to remove camera-distance and position effects. The implementation in `src/inference.py`:

```python
def _normalize(ff: np.ndarray) -> np.ndarray:
    raw   = ff.astype(np.float64)
    left  = raw[0:63].reshape(21, 3).copy()    # 21 joints × (x,y,z)
    right = raw[63:126].reshape(21, 3).copy()
    face  = raw[126:].reshape(10, 3).copy()    # 10 points × (x,y,z)

    # Hands: wrist (joint 0) becomes the origin
    for hand in [left, right]:
        if hand.any():
            hand -= hand[0]                             # shift: wrist → (0,0,0)
            scale = np.max(np.linalg.norm(hand, axis=1))
            if scale > 0:
                hand /= scale                           # scale: max reach → 1.0

    # Face: centroid becomes origin
    if face.any():
        face -= face.mean(axis=0)                       # shift: centroid → (0,0,0)
        scale = np.max(np.linalg.norm(face, axis=1))
        if scale > 0:
            face /= scale

    return np.concatenate([left.flatten(), right.flatten(), face.flatten()]).astype(np.float32)
```

**What this achieves:**
- A signer with large hands at close range and one with small hands far away now produce near-identical normalised vectors for the same sign
- Translation-invariant (position on screen doesn't matter)
- Scale-invariant (hand size doesn't matter)
- The sign shape itself — the relative positions of fingers — is preserved

### 4.2 Emotion Fusion

The model input is not just 156 landmark values — it is 163 values. The extra 7 are a **one-hot encoding** of the signer's detected emotion:

```python
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# Neutral emotion → one-hot vector:
[0, 0, 0, 0, 1, 0, 0]
#           ↑
#        index 4 = neutral
```

This is concatenated after the 156 normalised landmarks:
```
[landmark_0, ..., landmark_155,  angry, disgust, fear, happy, neutral, sad, surprise]
 ←────────────── 156 ──────────────────→  ←──────────── 7 ─────────────────→
```

**Why emotion matters for sign language:** The same hand shape can have different meanings depending on facial expression. For example, a question may be indicated by raised eyebrows. Emotion fusion lets the model theoretically use this information.

**Why it doesn't help much yet:** All training samples used `neutral` as the emotion label because we had no emotion ground-truth annotations. The model learned to mostly ignore the emotion slot. The architecture is ready — the upgrade path is to collect emotion-labelled data and retrain.

---

## 5. Data Augmentation

With ~10 videos per sign class, we need to artificially expand the training set. We apply 4 augmentation strategies (plus an optional rotation). Each produces a new `.npy` file saved alongside the original.

### 5.1 Horizontal Flip (`aug_hflip`)

Sign language is mostly symmetric — many signs have a mirrored equivalent, and even non-symmetric signs benefit from seeing the mirrored version.

```
Original:  [left_hand_63, right_hand_63, face_30]
Flipped:   [right_hand_63_mirrored, left_hand_63_mirrored, face_30_mirrored]
```

The x-coordinates are negated and hands swapped: left becomes right and vice versa.

### 5.2 Time Jitter (`aug_time_jitter`)

Signs performed at different speeds should still be recognised. We create slow (0.80×) and fast (1.20×) versions of each sequence by resampling the frame sequence.

```
Original 25 frames → resample to 20 frames (faster)
Original 25 frames → resample to 31 frames (slower)
```

### 5.3 Gaussian Noise (`aug_gaussian_noise`)

Adds small random perturbations (σ = 0.005) to all landmark coordinates. This prevents the model from memorising exact coordinate values and forces it to learn robust shape representations.

```
noisy_landmark = original_landmark + N(0, 0.005)
```

### 5.4 Landmark Dropout (`aug_landmark_dropout`)

Randomly zeros out 1–2 non-wrist joints per frame with probability 0.1. This simulates partial hand occlusion (one finger hidden behind another) and forces the model to classify correctly even with incomplete hand information.

### 5.5 Rotation (`aug_rotation_2d`, optional)

Rotates hand landmarks ±10° around the wrist in the XY plane. Face landmarks are not rotated. This simulates the signer tilting their wrist.

### 5.6 Data Leakage Guard for Augmentation

Augmented files for videos in the validation split are automatically excluded from training:

```python
# export_eval_arrays.py
for aug_suffix in ["_aug_flip", "_aug_slow", "_aug_fast", "_aug_noise"]:
    if video_stem + aug_suffix in val_video_stems:
        continue   # ← drop augmented val videos from train
```

---

## 6. Model Architecture

### 6.1 Model v1 — Baseline MLP (no augmentation)

Used only for ablation comparison. Shows how much augmentation improves accuracy.

```
Input:  163 features  (156 landmarks + 7 emotion, all zeros during training)
        ↓
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.3)
        ↓
Dense(128, activation='relu')
BatchNormalization()
Dropout(0.3)
        ↓
Dense(64, activation='relu')
        ↓
Dense(47, activation='softmax')   ← 47 sign classes
        ↓
Output: probability distribution over 47 classes
```

**File:** `artifacts/model_v1.keras` (369 KB)
**Val accuracy:** 81.82%

### 6.2 Model v2 — Augmented MLP + Emotion Fusion (PRODUCTION)

Same architecture as v1 but trained on augmented data (4× more training samples). This is the model used in all production deployments.

```
Input:  163 features
  [0:156]   normalised hand + face landmarks
  [156:163] emotion one-hot (all "neutral" during training)
        ↓
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.3)
        ↓
Dense(128, activation='relu')
BatchNormalization()
Dropout(0.3)
        ↓
Dense(64, activation='relu')
        ↓
Dense(47, activation='softmax')
        ↓
Output: (label, confidence, emotion)
```

**File:** `artifacts/model_v2.keras` (973 KB)
**Val accuracy:** 91.74%  |  **Macro F1:** 0.906

### 6.3 Model v3 — Augmented LSTM + Emotion Fusion

Designed for dynamic signs where motion trajectory matters. Instead of a single frame, reads a sequence of 30 consecutive frames.

```
Input:  (30, 163) sequence  — 30 frames, each 163 features
        ↓
Bidirectional LSTM(128)      ← reads sequence forward and backward
Dropout(0.3)
        ↓
Bidirectional LSTM(64)
Dropout(0.3)
        ↓
Dense(64, activation='relu')
        ↓
Dense(47, activation='softmax')
        ↓
Output: (label, confidence, emotion)
```

**File:** `artifacts/model_v3.keras` (5.3 MB)
**Val accuracy:** 90.91%  |  **Macro F1:** 0.894

The LSTM is slightly below v2 overall, but outperforms v2 on motion-dependent signs where the trajectory matters more than any single frame.

### 6.4 Model Comparison

| Model | Architecture | Augmentation | Val Acc | Macro F1 | Size | Latency |
|-------|-------------|-------------|---------|----------|------|---------|
| v1 | MLP | None | 81.82% | 0.801 | 369 KB | 487 ms |
| **v2** | **MLP** | **Yes** | **91.74%** | **0.906** | **973 KB** | **469 ms** |
| v3 | LSTM | Yes | 90.91% | 0.894 | 5.3 MB | 462 ms |

> Latency includes full MediaPipe Holistic extraction (~450ms) + model inference (~5ms).
> Model-only inference is approximately 2–5ms.

### 6.5 Why Not a Transformer or Attention Model?

With 121 validation samples across 47 classes, a Transformer would have far more parameters than training examples and would severely overfit. MLP and LSTM are appropriately sized for this dataset scale.

---

## 7. Training Process

Training was conducted in Google Colab (Phase 1 notebook). Key training parameters:

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 with ReduceLROnPlateau |
| Loss | Categorical crossentropy |
| Batch size | 32 |
| Epochs | up to 100 with EarlyStopping (patience=10) |
| Validation strategy | Video-level 80/20 split, augmented val sequences excluded |
| Random seed | 42 (reproducibility) |

**Why Adam:** Adaptive learning rate converges reliably on small datasets without extensive hyperparameter tuning.

**Why EarlyStopping:** Prevents overfitting — training stops when validation loss stops improving for 10 epochs, regardless of epoch limit.

**Why BatchNorm + Dropout:** BatchNorm stabilises activations and speeds convergence. Dropout (0.3) prevents co-adaptation of neurons, a form of regularisation needed given the limited data.

---

## 8. Evaluation Results

Evaluation is performed by `src/evaluate.py` using `artifacts/X_mlp_val.npy` (121 samples, 47 classes).

### 8.1 Ablation Table (`results/ablation_table.csv`)

| Model | Val Acc | Macro F1 | Mean Latency | Std Latency |
|-------|---------|----------|-------------|-------------|
| v1: Baseline MLP | 81.82% | 0.8015 | 487 ms | 95 ms |
| **v2: Aug MLP + Emotion** | **91.74%** | **0.9060** | **469 ms** | **92 ms** |
| v3: Aug LSTM + Emotion | 90.91% | 0.8943 | 462 ms | 94 ms |

**Key finding:** Data augmentation alone (v1 → v2) gives +9.92 percentage points of accuracy improvement. This is the biggest single improvement in the project.

### 8.2 Output Files

| File | Contents |
|------|---------|
| `results/ablation_table.csv` | Full model comparison with metrics |
| `results/confusion_matrix.png` | 47×47 heatmap — which signs get confused |
| `results/classification_report.txt` | Per-class precision, recall, F1 |
| `results/failure_analysis.png` | Top 3 confused sign pairs with visualisation |

### 8.3 Latency Breakdown

The 469 ms end-to-end latency is dominated by MediaPipe Holistic, not the model:

| Component | Time |
|-----------|------|
| MediaPipe Holistic extraction | ~450 ms |
| Feature normalisation | < 1 ms |
| Emotion one-hot construction | < 1 ms |
| MLP v2 inference | ~5 ms |
| Sliding window logic | < 1 ms |
| **Total** | **~469 ms** |

> This is the **evaluate.py** measurement latency (warm-up + full pipeline on CPU).
> In real-time demo mode with frame skipping (`FRAME_SKIP=3`), the display runs at 15–30 FPS because inference runs every 3rd frame and caches the result for the other 2.

---

## 9. Inference Engine — predict_frame()

The entire inference logic lives in `src/inference.py`. This file is the core contract — all other files (demo, web server, evaluation, mobile) depend on it.

### 9.1 The Complete Pipeline

```
predict_frame(frame: np.ndarray, cached_emotion="neutral")
│
├─ Frame skipping check
│   If FRAME_SKIP=3 and this frame index % 3 ≠ 0:
│   └─ Return cached_prediction from last compute frame (no inference)
│
├─ Resize frame to 320×240 (speed)
│
├─ MediaPipe Holistic → extract 156-dim raw landmark vector
│   If left hand missing  → fill [0:63]   with 0.0
│   If right hand missing → fill [63:126] with 0.0
│   If face missing       → fill [126:156] with 0.0
│
├─ Hand detection check
│   If raw[0:126].sum() == 0 (no hands detected):
│   └─ Return ("No hand detected", 0.0, emotion)
│
├─ Activation gate
│   velocity = ||current_hand_landmarks - previous_hand_landmarks||
│   If velocity < VELOCITY_THRESHOLD=0.02 (or first frame):
│   └─ Return ("Ready", 0.0, emotion)   ← hands are still
│
├─ Post-commit cooldown
│   If _cooldown_remaining > 0:
│   │   Decrement counter
│   └─ Return ("Detecting", 0.0, emotion)   ← too soon after last commit
│
├─ Normalise landmarks → 156-dim normalised vector
│
├─ Construct 163-dim feature vector
│   [normalised_landmarks_156, emotion_one_hot_7]
│
├─ MLP forward pass
│   probs = model(features)    → shape (47,)
│   class_idx = argmax(probs)
│   confidence = probs[class_idx]
│   label = idx2label[class_idx]
│
├─ Sliding window
│   _pred_window.append((label, confidence))   ← deque(maxlen=5)
│   If len(_pred_window) == 5:
│     top = most frequent label in window
│     top_confs = [conf for (lbl, conf) in window if lbl == top]
│     If count(top) >= MIN_VOTES=3 AND mean(top_confs) >= MIN_CONFIDENCE=0.60:
│       _pred_window.clear()
│       _cooldown_remaining = COMMIT_COOLDOWN=8
│       └─ Return (top, mean(top_confs), emotion)   ← COMMIT
│
└─ Return ("Detecting", confidence, emotion)   ← still accumulating
```

### 9.2 The Activation Gate

**Why:** The model runs ~30 times per second. If the signer holds their hands still between signs, we'd run inference on the same static frame 30 times, producing 30 identical predictions that fill the sliding window and commit a false sign.

**How:** We measure the L2 norm of the difference between the current frame's hand landmarks and the previous frame's. If it's below `VELOCITY_THRESHOLD = 0.02`, the hands are considered still and we skip inference for that frame.

```python
velocity = ||hands_current - hands_previous||
if velocity < 0.02:
    return ("Ready", 0.0, emotion)   # skip
```

**First frame:** On the very first frame after startup (or after `reset_gate()`), `_lm_prev` is `None`. We return `False` (not moving) and save the current landmarks as reference. This is the correct behaviour — we need a baseline before we can measure velocity.

### 9.3 The Sliding Window

**Why:** A single frame can be misclassified. The signer's hand might be mid-transition between signs. Requiring 3 out of 5 consecutive frames to agree eliminates most transient false positives.

**The old bug (now fixed):** The confidence check used `np.mean(ALL 5 confs)`. If 2 of the 5 frames predicted the wrong label at low confidence (e.g., 0.3), they dragged the average below 0.60, preventing commit even when the 3 correct votes had confidence 0.70+. The window would slide forever without committing.

**The fix:** Only average the confidence of the frames that voted for the winning label:
```python
top_confs = [c for (lbl, c) in window if lbl == top]
# If top got 3/5 votes, top_confs has 3 values, not 5.
```

### 9.4 Post-Commit Cooldown

**Why:** After a word is committed and `_pred_window.clear()` is called, if the signer is still holding the same hand shape, the window immediately refills with the same sign and commits it again within 5 frames. This looks like the model is "stuck" on the first word.

**How:** After every commit, `_cooldown_remaining = COMMIT_COOLDOWN = 8`. For the next 8 active compute-frames, the activation gate passes but we return "Detecting" without appending to the window. This gives the signer ~0.8–1.0 seconds to transition to the next sign before the window starts accumulating again.

### 9.5 Frame Skipping

**Why:** Running full MediaPipe + model inference on every frame at 30fps would require ~469ms × 30 = too slow. But showing a prediction on every frame (even a cached one) keeps the UI smooth.

**How:** We run full inference every `FRAME_SKIP = 3` frames and reuse the cached prediction for the other 2:
```
Frame 1: full inference  → label="HELLO", cache it
Frame 2: return cache    → "HELLO" (no compute)
Frame 3: return cache    → "HELLO" (no compute)
Frame 4: full inference  → update cache
...
```

**Web server exception:** The browser sends 10 fps (not 30 fps). With FRAME_SKIP=3, only 3.3 fps would get inference — the window would take 1.5 seconds to fill. So the web server calls `predict_frame(..., disable_frame_skip=True)` to get inference on every received frame.

### 9.6 Sentence Word Queue

After the sliding window commits a word, it's appended to `sentence_words = deque(maxlen=8)`. This accumulates a running sentence visible at the top of the screen.

Deduplication prevents the same word from appearing twice in a row (the cooldown already prevents rapid re-commit, but this is a second safety net at the display layer).

---

## 10. Desktop Application

### 10.1 Three-Thread Architecture

The desktop app (`demo.py`) uses three parallel threads to keep the camera feed smooth even when inference is slow.

```
┌─────────────────────────────────────────────────────────────────┐
│  Thread 1 — Capture (T1-Capture)                                │
│  OpenCV VideoCapture → frame_queue (maxsize=2)                  │
│  Runs at camera FPS. Drops old frames if queue is full.         │
└───────────────────┬─────────────────────────────────────────────┘
                    │ frame
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Thread 2 — Inference (T2-Inference)                            │
│  predict_frame() → result_queue (maxsize=2)                     │
│  Runs at ~10fps (inference rate, not camera rate).              │
│  Pushes: (frame, label, conf, emotion, frame_count, mp_results) │
└───────────────────┬─────────────────────────────────────────────┘
                    │ (frame, label, conf, ...)
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Main Thread — Display                                           │
│  Reads result_queue → draws overlays → cv2.imshow()            │
│  Keyboard: Q=quit, L=landmarks, D=debug, R=reset, C=clear      │
└─────────────────────────────────────────────────────────────────┘
        ↑
        │ emotion update (every DEEPFACE_INTERVAL=5 frames)
┌───────┴─────────────────────────────────────────────────────────┐
│  Thread 3 — Emotion (T3-Emotion)                                │
│  DeepFace.analyze() on latest frame → updates _cached_emotion   │
│  Protected by threading.Lock (emotion_lock).                    │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Queue Design

Both queues have `maxsize=2`. If a queue is full when a producer tries to push, it evicts the oldest item first. This prevents ever-growing lag:

```python
if result_queue.full():
    try:
        result_queue.get_nowait()   # evict oldest
    except queue.Empty:
        pass
result_queue.put(new_result)
```

If inference is slow, the display thread shows a slightly stale result but never falls behind by more than 2 frames.

### 10.3 Label Display Logic

The display thread has a smart filter: "Detecting" is not shown as the main label. Instead, the last committed word stays visible while the model is detecting the next sign:

```python
if label not in ("Detecting", "Error"):
    last_label = label   # update display
    if label not in STATE_COLOURS:   # it's a real sign
        if label != last_committed:
            sentence_words.append(label)
            last_committed = label
```

So the screen shows: `HELLO` → `HELLO` (while detecting next) → `THANKS`
Not: `HELLO` → `Detecting...` → `Detecting...` → `Detecting...` → `THANKS`

### 10.4 Sprint Mode (Alternative Architecture)

Sprint mode (`demo.py --sprint`) is a simpler single-loop alternative:

```
while True:
    frame = camera.read()
    active, raw_landmarks = gate_from_frame(frame)
    if not active:
        show "Ready"
    else:
        lbl, conf = predict_frame(frame, raw=True)   # per-frame, no sliding window
        committed = _sprint_smooth(pred_window, lbl, conf)   # app-layer smoothing
        if committed:
            pred_window.clear()
            show committed sign
```

The smoothing logic mirrors the inference.py sliding window but lives in the app layer. It uses `SPRINT_MIN_CONFIDENCE = 0.65` (slightly higher than the main mode's 0.60) because `raw=True` gives per-frame predictions without the window's noise averaging.

### 10.5 Keyboard Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| L | Toggle landmark skeleton overlay |
| D | Toggle debug panel (top-5 class probabilities) |
| R | Reset prediction window and activation gate |
| C | Clear the sentence word queue |

---

## 11. Web Application

### 11.1 FastAPI WebSocket Server (`web/server.py`)

```
Browser / Mobile PWA
        │  WebSocket  (base64 JPEG frame every 100ms)
        ▼
FastAPI  /ws  endpoint
        │
        ├─ base64 decode → JPEG bytes
        ├─ cv2.imdecode → BGR numpy array
        └─ predict_frame(frame, "neutral", disable_frame_skip=True)
                │
                └─ JSON response:
                   {
                     "label":       "HELLO",
                     "confidence":  0.89,
                     "emotion":     "neutral",
                     "latency_ms":  45.2,
                     "fps_hint":    22.1
                   }
```

**Why `disable_frame_skip=True`:** The browser sends 10 fps. With `FRAME_SKIP=3`, only 3.3 fps would reach the model. The sliding window needs 5 compute frames to fill, so commits would take 1.5 seconds. With frame skip disabled, every frame is processed and commits happen in 0.5 seconds (5 frames × 100ms).

**Why `cached_emotion="neutral"`:** DeepFace requires a 500MB download and crashes on frames without faces. For the web server, we always pass neutral — this is architecturally correct and documented.

### 11.2 Browser Frontend (`web/index.html`)

A Progressive Web App (PWA) — installable on Android/iOS home screen from the browser, no app store.

```javascript
// Camera → canvas → base64 → WebSocket (every 100ms)
setInterval(() => {
    ctx.drawImage(video, 0, 0, 320, 240);
    const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    ws.send(b64);
}, 100);

// Receive predictions → update UI
ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    // Update status, confidence bar, emotion
    // If real sign (not meta-state): append to committedWords[]
};
```

**Sentence queue in browser:** `committedWords` array (max 10), deduped by `lastCommitted`. Meta-states (`Ready`, `Detecting`, `No hand detected`, `Error`) are filtered out — only real sign words accumulate. "Clear sentence" button resets the array.

### 11.3 Health Endpoint

```
GET /health
→ {"status": "ok", "model_loaded": true}
```

Used by monitoring tools and by the mobile app to wake up a sleeping Railway container before the demo.

---

## 12. Mobile Deployment — TFLite

### 12.1 What TFLite Is

TensorFlow Lite (TFLite) is a format for running TensorFlow/Keras models on mobile devices without a Python runtime or TensorFlow installation. The `.tflite` file is a compiled, optimised binary of the model.

| Property | Keras (`.keras`) | TFLite (`.tflite`) |
|----------|------------------|--------------------|
| Runs on | Desktop/server (Python) | Android, iOS, embedded |
| Size | 973 KB | ~400 KB (dynamic quantized) |
| Needs server? | Yes | **No — on-device** |
| Inference time | ~5ms (model only) | ~2ms on modern phones |
| Python required? | Yes | No |

### 12.2 Exporting TFLite

```bash
# Export with validation (recommended)
python scripts/prepare_mobile.py

# Or export only
python scripts/export_tflite.py --validate

# For exact numerical parity (larger file)
python scripts/export_tflite.py --optimize float32 --validate
```

**Dynamic quantization** (default): converts weights from float32 to int8 during inference. 2–5× faster, ~half the file size, <0.1% accuracy drop.

### 12.3 Mobile Integration Spec

**Input tensor:** shape `[1, 163]`, dtype `float32`

```
[  0 :  63]  Left hand   (21 joints × x,y,z)  — zeros if not detected
[ 63 : 126]  Right hand  (21 joints × x,y,z)  — zeros if not detected
[126 : 156]  Face        (10 points × x,y,z)  — zeros if not detected
[156 : 163]  Emotion     [0, 0, 0, 0, 1, 0, 0] — always this (neutral)
```

**Output tensor:** shape `[1, 47]`, dtype `float32` — softmax probabilities.

**CRITICAL:** The old model likely expected 156-dim input. The new model expects 163. To update:

```kotlin
// Kotlin (Android)
val features = FloatArray(163)
landmarks.copyInto(features, 0, 0, 156)   // 156 landmark values
features[160] = 1.0f                        // neutral emotion index 4
```

```swift
// Swift (iOS)
var features = [Float](repeating: 0, count: 163)
features.replaceSubrange(0..<156, with: landmarks)
features[160] = 1.0   // neutral emotion index 4
```

### 12.4 Normalization (Must Match Exactly)

The model was trained on normalised landmarks. The mobile app **must** apply the same normalisation or accuracy degrades significantly.

**For each hand:**
1. Get wrist position = joint[0]
2. Subtract wrist from all 21 joints
3. Compute max Euclidean distance from wrist across all joints
4. If max > 0: divide all joints by max

**For face:**
1. Compute centroid = mean of all 10 points
2. Subtract centroid from all 10 points
3. Compute max Euclidean distance from centroid
4. If max > 0: divide all points by max

### 12.5 Face Landmark Indices

MediaPipe outputs 478 face landmarks. We use only these 10:
```
[0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
```
These are: nose tip, between eyebrows, mouth corners, chin, eye corners.

### 12.6 Recommended Sliding Window on Mobile

```
WINDOW_SIZE     = 5
MIN_VOTES       = 3    (3 of 5 frames must agree)
MIN_CONFIDENCE  = 0.60 (mean conf of agreeing frames)
COMMIT_COOLDOWN = 8    (frames to skip after commit)
```

After commit: clear the window, skip 8 frames before accumulating again.

---

## 13. Cloud Deployment — Railway

### 13.1 Architecture

```
Developer pushes to GitHub main branch
        │
        ▼
GitHub Actions (.github/workflows/deploy.yml)
        │  railway up --service esl-server --detach
        ▼
Railway builds Docker image (Dockerfile)
        │
        ├─ FROM python:3.11-slim
        ├─ Install system libs (libGL, libglib)
        ├─ pip install -r requirements.txt
        ├─ python scripts/download_model.py  ← downloads from HuggingFace Hub
        └─ CMD uvicorn web.server:app --workers 1
        │
        ▼
Public URL: wss://esl-xxxxx.up.railway.app/ws
        │
        ▼
Browser or web-connected mobile app sends frames, receives JSON
```

### 13.2 Why HuggingFace Hub for Model Storage

Git repositories have a 100MB file size limit. `model_v2.keras` is 973KB (fine), but `holistic_landmarker.task` is 13.7MB and LSTM arrays can reach 34MB. HuggingFace Hub is free, permanent, versioned storage designed for ML artifacts. The Dockerfile downloads them at build time — downloaded once and cached in the Docker layer.

### 13.3 Why `--workers 1`

MediaPipe Holistic creates OS-level handles to the `.task` model file and internal state that is not safe to share across forked processes. `uvicorn --workers 2+` would fork the process, causing crashes. A single async worker handles all concurrent WebSocket connections — this is sufficient for demo load since each connection is sequential (send frame → wait → receive result).

### 13.4 Promoting a New Model

```bash
# 1. Upload new model to HuggingFace
python scripts/promote_model.py model_v3.keras

# 2. Trigger Railway rebuild (empty commit)
git commit --allow-empty -m "deploy: promote model_v3"
git push

# Railway downloads new model at build time, redeploys automatically
```

---

## 14. Bug History and Fixes

### Data Pipeline (Phase 1)
- **Fixed:** Data leakage from frame-level train/val split. Changed to video-level split — no frame from a validation video appears in training.

### Inference Engine (Phase 2 / Session 2026-06-09)

| Bug | Symptom | Fix |
|-----|---------|-----|
| `np.mean(ALL confs)` in sliding window | Model stuck after first word — window never commits second sign because noise frames drag mean below threshold | Changed to `top_confs = [c for lbl,c in window if lbl==top]` |
| No post-commit cooldown | Same word immediately re-commits in next 5 frames — appears stuck | Added `COMMIT_COOLDOWN = 8` active frames after each commit |
| `FRAME_SKIP=3` active in web server | Window takes 1.5s to fill at 10fps ÷ 3 = 3.3fps | Added `disable_frame_skip=True` in server call |
| `landmark_gate.py` returns `True` on first call | Sprint mode processes a garbage velocity on frame 1 | Changed to `return False` — matches `inference.py` behaviour |
| `_sprint_smooth` never clears window | Old sign frames contaminate next sign's vote | Added `pred_window.clear()` after commit |
| `_sprint_smooth` uses `np.mean(ALL confs)` | Same noise-averaging bug as main mode | Changed to use `top_confs` |
| `_frame_ts_ms` not thread-safe | Non-monotonic timestamps possible under concurrent calls | Added `_frame_ts_lock = threading.Lock()` |

### Deployment (2026-06-09)
- **Fixed:** Dockerfile had unresolved git merge conflict markers — image could not be built at all.
- **Fixed:** `promote_model.py` passed raw `bytes` to `HfApi.upload_file()` — wrapped in `BytesIO`.
- **Fixed:** Dead code in `server.py`: `lbl if lbl not in (...) else lbl` — simplified to `lbl`.

---

## 15. Viva Q&A — Prepared Answers

**"Why landmark-based instead of CNN on raw images?"**
> With approximately 10 videos per sign class, a CNN would overfit severely — it needs thousands of images. MediaPipe gives us 156 clean numbers representing every hand joint position, stripped of background and lighting variation. Our MLP is 973KB, trains in minutes, and runs at real-time on CPU. A CNN would be 14+MB, need a GPU, and still perform worse at this dataset size.

**"What is model accuracy and how did you measure it?"**
> Model v2 achieves 91.74% accuracy and 0.906 macro F1 on a held-out validation set of 121 samples across 47 sign classes. We split at the video level — 80% of videos per class to training, 20% to validation. No frame from a validation video appears in training, including augmented versions.

**"What is the data leakage bug you fixed?"**
> Frame-level train/val splitting lets multiple frames from the same video appear in both sets. Those frames share the signer's hand proportions, so the model memorises the person not the sign — artificially inflated accuracy. We fixed this by splitting at the video level: 80% of videos go to train, 20% to val. No video appears in both sets.

**"Why did augmentation improve accuracy so much (+9.9%)?"**
> With 10 videos per class, the model sees almost no variety — one signer, one angle, one speed. Augmentation synthetically creates: mirrored signs (flip), speed variations (time jitter ×0.80 and ×1.20), noise tolerance (Gaussian noise σ=0.005), and partial occlusion robustness (landmark dropout). This 4× expansion of the training set substantially improves generalisation.

**"Why LSTM for some signs and MLP for others?"**
> Signs fall into two categories: static (where meaning comes from hand shape alone — e.g., letters) and dynamic (where meaning comes from motion trajectory — e.g., directional signs). An MLP classifies a single normalised frame and works well for static signs. An LSTM reads 30 consecutive frames and captures the motion arc that distinguishes dynamic signs. Our ablation table shows LSTM (90.91%) is competitive with MLP (91.74%) overall, with LSTM specifically outperforming on motion-dependent signs.

**"Why is emotion feature not improving accuracy?"**
> We trained all models with `neutral` as the emotion label for every training sample because our dataset has no emotion ground-truth annotations. The model learned to mostly ignore the 7-dimensional emotion input. The architecture supports it — the slot is in the input — but the model hasn't learned to use variation there. The upgrade path: collect emotion-labelled signing data, retrain model_v4. Zero architecture changes needed.

**"How does sliding window prevent false positives?"**
> We require 3 of the last 5 consecutive inference frames to agree on the same label AND the mean confidence of those agreeing frames must exceed 0.60. After committing a word, we enforce an 8-frame cooldown (approximately 0.8 seconds) where the window does not accumulate, preventing the same sign from immediately re-triggering. A single misclassified frame cannot trigger a commit.

**"What are limitations?"**
> Three main limitations: (1) **Single signer in training** — the model has seen signs from one person, so accuracy drops on unseen signers with different hand proportions or signing style; (2) **47 of 55 sign vocabulary** — our MVP covers 47 Egyptian signs; (3) **Isolated sign recognition only** — we classify individual words, not connected sentences. For connected speech you would need a language model on top of our sign classifier. All three are documented with a clear improvement roadmap.

**"Why not a native mobile app?"**
> We provide a PWA (Progressive Web App) — installable on Android and iOS home screen via a browser, no app store, no native SDK, no cross-compilation. The same `predict_frame()` function powers all three interfaces: desktop app, browser frontend, and via TFLite — the mobile app. For native deployment, the upgrade path is TFLite + MediaPipe iOS/Android SDKs.

**"What is TFLite and why use it?"**
> TFLite (TensorFlow Lite) is a compiled, quantized format of a Keras model that runs on mobile devices without Python or TensorFlow. Our 973KB Keras model converts to ~400KB TFLite with dynamic quantization. The mobile app runs the entire pipeline on-device — MediaPipe for landmark extraction, normalization, and TFLite inference — without any network connection. This is faster (2ms model inference), more reliable (works offline), and more scalable than the WebSocket server approach.

**"How would scale this to all 55 signs and multiple signers?"**
> The architecture needs no changes — only the training data. Scaling requires: (1) recording data from 4+ signers (adds ±15–25% accuracy on unseen people), (2) recording all 55 sign classes with 10+ videos each, (3) rerunning the Phase 1 Colab notebook, (4) running `scripts/promote_model.py` to push the new model. The CI/CD pipeline automatically rebuilds the Railway container with the new model — zero downtime.

---