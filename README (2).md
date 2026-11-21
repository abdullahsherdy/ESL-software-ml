<div align="center">

# ESL-software-ml

### Real-Time Egyptian Sign Language Recognition

Recognize Egyptian Sign Language (ESL) from a webcam, a video file, or a phone browser, and show the sign, a confidence score, and the signer's emotion instantly, with no cloud services and no human interpreter.

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-0097A7)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Screenshots](#screenshots)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Team](#team)
- [License](#license)

---

## About the Project

Communication between Deaf and hard-of-hearing people and the hearing community often depends on interpreters or written text, and interpreters are not always available.

**ESL-software-ml** is a graduation project that acts as a technical assistant. It reads Egyptian Sign Language from live video and returns:

| Output | Description |
|---|---|
| **Sign label** | The recognized sign (e.g. `HELLO`, `THANKS`) |
| **Confidence** | A score from 0.0 to 1.0 |
| **Emotion** | The signer's facial emotion (desktop full mode) |

Everything runs **locally**, so it is fast, private, and works without an internet connection.

---

## Features

- **Real-time recognition** from webcam or video file
- **Landmark-based pipeline**: MediaPipe Holistic extracts hand and face keypoints, so no heavy image model runs per frame
- **Three trained models**: baseline MLP, augmented MLP (default), and Bi-LSTM
- **Emotion awareness** using DeepFace on a separate thread
- **Motion gate**: predicts only when the hands are moving, which cuts false detections
- **Temporal smoothing**: sliding-window voting keeps labels from flickering
- **Mobile-friendly web app** (FastAPI + WebSocket, PWA-ready) that works from a phone browser
- **Stable API**: `predict_frame()` returns `(label, confidence, emotion)` for easy integration
- **Built-in evaluation**: ablation table, confusion matrix, and failure analysis
- **Docker support**

---

## Screenshots

> Add your app screenshots to a `docs/screenshots/` folder and update the paths below.

| Home / Camera | Live Recognition | Result + Emotion |
|:---:|:---:|:---:|
| ![Home](docs/screenshots/home.png) | ![Live](docs/screenshots/live.png) | ![Result](docs/screenshots/result.png) |

---

## How It Works

```
Camera / Video
     │
     ▼
┌──────────────┐   ┌────────────────────┐   ┌──────────────┐   ┌───────────────┐
│ Capture      │ → │ MediaPipe Holistic │ → │ Normalize +  │ → │ Keras model   │
│ (OpenCV)     │   │ 156 landmarks      │   │ emotion (163)│   │ (MLP / LSTM)  │
└──────────────┘   └────────────────────┘   └──────────────┘   └───────┬───────┘
                                                                       ▼
                         Display  ◄──  Sliding-window voting  ◄──  Softmax
                   (sign · confidence · emotion · FPS)
```

1. Each frame is resized to 320×240 and passed to MediaPipe Holistic.
2. The system extracts **156 landmark values**: 21 joints per hand × 2, plus 10 face points.
3. The values are normalized and joined with a **7-value emotion one-hot**, giving a **163-dimension** feature vector.
4. If the hands are still, the app shows `Ready`. If they are moving, the model predicts.
5. A 5-frame voting window commits a label only after **at least 3 votes** and **confidence ≥ 0.65**.

In full desktop mode the work is split across three threads (capture, inference, emotion) so the UI stays responsive.

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.10 / 3.11 |
| Computer vision | OpenCV, MediaPipe Holistic (Tasks API) |
| Machine learning | TensorFlow / Keras (MLP, Bi-LSTM) |
| Emotion detection | DeepFace |
| Web backend | FastAPI, WebSocket, Uvicorn |
| Web frontend | HTML, JavaScript (`getUserMedia`), PWA manifest |
| DevOps | Docker, GitHub Actions |

---

## Getting Started

### Prerequisites

- Python **3.10 or 3.11**
- A webcam, or a video file
- Model files in `artifacts/`: `model_v2.keras`, `label2idx.json`, `holistic_landmarker.task`

### Installation

**Windows (PowerShell)**

```powershell
git clone https://github.com/abdullahsherdy/ESL-software-ml.git
cd ESL-software-ml
.\setup_venv.ps1
```

**Linux / macOS**

```bash
git clone https://github.com/abdullahsherdy/ESL-software-ml.git
cd ESL-software-ml
./setup_venv.sh
```

**Manual setup**

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note:** DeepFace needs `tf-keras` with TensorFlow 2.16+, and it downloads about 500 MB of weights on first run. If you hit `jax` / `ml-dtypes` conflicts, uninstall `jax` and `jaxlib` from the environment.

### Quick check (no camera needed)

```bash
python scripts/smoke_check.py
```

---

## Usage

### Desktop app: full mode (with emotion)

```bash
python demo.py                      # default webcam
python demo.py --source 0           # choose a camera
python demo.py --source video.mp4   # use a video file
```

### Desktop app: sprint mode (lightweight, no DeepFace)

```bash
python demo.py --sprint
```

### Web app

```bash
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
# or: ./run_web.sh   |   .\run_web.ps1
```

Open `http://localhost:8000`. To use it from your phone, connect it to the same Wi-Fi network and open `http://<your-computer-ip>:8000`.

> Mobile browsers usually require HTTPS before they allow camera access, except on `localhost`. Use a secure tunnel or a reverse proxy for phone testing.

### Keyboard shortcuts (desktop)

| Key | Action |
|:---:|---|
| `Q` | Quit |
| `L` | Toggle hand skeleton |
| `D` | Debug mode |
| `R` | Reset voting window |

### Use it in your own code

```python
from src.inference import load_model, predict_frame

load_model()
label, confidence, emotion = predict_frame(frame)   # frame = BGR image from OpenCV
```

---

## Models

| Model | Architecture | Input | Notes |
|---|---|---|---|
| `model_v1` | MLP | `(N, 163)` | Baseline, no augmentation |
| **`model_v2`** | MLP | `(N, 163)` | **Default**: trained with augmentation |
| `model_v3` | Bi-LSTM | `(N, 30, 163)` | Temporal model, 30-frame sequences |

Training augmentation includes horizontal flip, time jitter, Gaussian noise, and landmark dropout.

---

## Evaluation

Run the full evaluation to compare all three models:

```bash
python -m src.evaluate
```

Outputs are saved to `results/`:

- `ablation_table.csv` — validation accuracy, macro F1, and latency per model
- `confusion_matrix.png`
- `classification_report.txt`
- `failure_analysis.png`

| Model | Val Acc | Macro F1 | Latency mean (ms) | Latency std (ms) |
|---|---:|---:|---:|---:|
| v1: Baseline MLP (no aug) | 0.8182 | 0.8015 | 487.47 | 94.83 |
| v2: Aug MLP + Emotion | 0.9174 | 0.906 | 468.96 | 91.51 |
| v3: Aug LSTM + Emotion | 0.9091 | 0.8943 | 461.81 | 94.46 |

---

## Project Structure

```
ESL-software-ml/
├── demo.py               # Desktop application
├── src/
│   ├── inference.py      # Real-time inference + emotion cache
│   ├── augmentation.py   # Training data augmentation
│   ├── evaluate.py       # Model evaluation
│   ├── paths.py          # Artifact path resolution
│   └── landmark_gate.py  # Motion gate (sprint mode)
├── web/                  # FastAPI + WebSocket web app
├── artifacts/            # Trained models and label map
├── results/              # Evaluation outputs
├── scripts/              # Helper scripts
├── data/                 # Dataset
├── Dockerfile
├── requirements.txt
└── PROJECT_DOCS.md       # Detailed technical documentation
```

---

## Documentation

- [`PROJECT_DOCS.md`](PROJECT_DOCS.md): full technical documentation (architecture, feature vector, threading, design decisions)
- [`ZERO_TO_HERO.md`](ZERO_TO_HERO.md): beginner-friendly walkthrough

---

## Team

| Name | Role |
|---|---|
| Abdullah Sherdy | ML / Software |
| *Add teammate* | *Role* |

**Supervisor:** *Add supervisor name*
**University / Faculty:** *Add details*

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Built to make communication more accessible.

</div>
