"""
src/inference.py
================
predict_frame() interface contract

Interface contract (must never change shape without notifying Software team):
    predict_frame(frame: np.ndarray) -> tuple[str, float, str]
    Returns: (label, confidence, emotion)

Software team: during Phase 2 setup, replace with the stub at the bottom of
this file until the real model is in place.
"""

import numpy as np
import cv2
import threading
from collections import deque

# ── Optional heavy imports — deferred until load_model() is called ────────────
_tf      = None
_mp      = None
_DeepFace = None

# ── Constants ─────────────────────────────────────────────────────────────────
# NOTE: FACE_IDX order MUST match Phase 1 Cell 3 / Cell 8 exactly. Reordering
# these values silently shuffles positions 126-155 of the feature vector and
# makes the model predict against a distribution it never saw at training time.
FACE_IDX         = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
FEATURE_DIM      = 156
EMOTION_CLASSES  = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_DIM      = 7
NEUTRAL_IDX      = EMOTION_CLASSES.index("neutral")  # 4

MODEL_PATH       = "artifacts/model_v2.keras"
LABEL2IDX_PATH   = "artifacts/label2idx.json"

VELOCITY_THRESHOLD = 0.02   # tune empirically; increase if too many false activations
WINDOW_SIZE        = 5
MIN_VOTES          = 3
MIN_CONFIDENCE     = 0.65
DEEPFACE_INTERVAL  = 5      # run DeepFace every K frames (Thread 3)

# Semantic confidence modifiers for sign/emotion conflicts
EMOTION_CONFLICTS = {
    ("happy",   "angry"):   0.75,
    ("happy",   "disgust"): 0.75,
    ("sad",     "happy"):   0.75,
}

# ── Module-level state ────────────────────────────────────────────────────────
_model        = None
_idx2label    = None
_holistic     = None
_lm_prev      = None
_pred_window  = deque(maxlen=WINDOW_SIZE)

_cached_emotion = "neutral"
_emotion_lock   = threading.Lock()

# Most recent MediaPipe Holistic results. Written inside _extract_landmarks()
# (called by predict_frame in the inference thread). Read by the display
# thread via get_last_mp_results() so we do not have to re-run Holistic
# just to draw the skeleton overlay.
_last_mp_results = None

_model_loaded = False


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH, label_path: str = LABEL2IDX_PATH) -> None:
    """
    Load the Keras model, label map, and initialise MediaPipe Holistic.
    Must be called once before predict_frame().
    """
    global _tf, _mp, _DeepFace
    global _model, _idx2label, _holistic, _model_loaded

    import tensorflow as tf
    import mediapipe as mp
    _tf = tf
    _mp = mp

    _model    = tf.keras.models.load_model(model_path)
    import json
    with open(label_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    _idx2label = {v: k for k, v in label2idx.items()}

    # ── Input-shape sanity check ──────────────────────────────────────────────
    # predict_frame() always concatenates FEATURE_DIM landmark + EMOTION_DIM
    # one-hot = 163 values. If someone points MODEL_PATH at model_v1.keras
    # (156-dim, no emotion) or at the LSTM variant, predict_frame() will throw
    # a shape mismatch deep inside model.predict(). Fail fast here instead.
    expected_dim = FEATURE_DIM + EMOTION_DIM
    actual_dim   = _model.input_shape[-1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"[inference] Model input dim mismatch: expected {expected_dim} "
            f"(= FEATURE_DIM {FEATURE_DIM} + EMOTION_DIM {EMOTION_DIM}) "
            f"but {model_path} has input_shape={_model.input_shape}. "
            f"Point MODEL_PATH at model_v2.keras (MLP + emotion)."
        )

    # static_image_mode=False → optimised for video streams (Phase 2)
    _holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    _model_loaded = True
    print(f"[inference] Model loaded — {len(_idx2label)} classes | input dim: {_model.input_shape}")


def update_emotion_async(frame: np.ndarray) -> None:
    """
    Called by Thread 3 every DEEPFACE_INTERVAL frames.
    Updates the module-level cached emotion safely.
    """
    global _DeepFace, _cached_emotion
    if _DeepFace is None:
        try:
            from deepface import DeepFace as _df
            _DeepFace = _df
        except ImportError:
            return

    try:
        result  = _DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        emotion = (result[0]["dominant_emotion"]
                   if isinstance(result, list)
                   else result["dominant_emotion"])
        with _emotion_lock:
            _cached_emotion = emotion
    except Exception:
        # DeepFace can time out or fail on a bad frame — keep last cached value
        pass


def predict_frame(frame: np.ndarray) -> tuple:
    """
    Main interface contract.

    Args:
        frame: BGR numpy array (from cv2.VideoCapture or WebSocket decode)
    Returns:
        (label: str, confidence: float, emotion: str)

    States returned in label:
        "No hand detected"  — hand not visible
        "Ready"             — hand still, waiting for motion
        "Detecting"         — motion detected but window not yet committed
        "<SIGN_LABEL>"      — committed prediction
    """
    global _lm_prev

    if not _model_loaded:
        raise RuntimeError("Call load_model() before predict_frame()")

    # ── 1. Resize for speed ───────────────────────────────────────────────────
    small = cv2.resize(frame, (320, 240))

    # ── 2. Extract landmarks ──────────────────────────────────────────────────
    raw_lm = _extract_landmarks(small)
    if raw_lm[:126].sum() == 0:
        return ("No hand detected", 0.0, "neutral")

    # ── 3. Activation gate ────────────────────────────────────────────────────
    moving = _activation_gate(raw_lm)
    if not moving:
        with _emotion_lock:
            emo = _cached_emotion
        return ("Ready", 0.0, emo)

    # ── 4. Normalise ──────────────────────────────────────────────────────────
    norm_lm = _normalize(raw_lm)

    # ── 5. Attach emotion vector ──────────────────────────────────────────────
    with _emotion_lock:
        emotion_str = _cached_emotion
    features = np.concatenate([norm_lm, _emotion_to_onehot(emotion_str)]).reshape(1, -1)

    # ── 6. Model inference ────────────────────────────────────────────────────
    probs      = _model(features, training=False)[0].numpy()
    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    label      = _idx2label[class_idx]

    # ── 7. Sliding window majority vote ───────────────────────────────────────
    _pred_window.append((label, confidence))
    if len(_pred_window) == WINDOW_SIZE:
        labels = [p[0] for p in _pred_window]
        confs  = [p[1] for p in _pred_window]
        top    = max(set(labels), key=labels.count)
        if (labels.count(top) >= MIN_VOTES
                and np.mean(confs) >= MIN_CONFIDENCE):
            mod = EMOTION_CONFLICTS.get((top.lower(), emotion_str.lower()), 1.0)
            _pred_window.clear()
            return (top, float(np.mean(confs)) * mod, emotion_str)

    return ("Detecting", float(confidence), emotion_str)


def get_holistic():
    """Return the MediaPipe Holistic instance (for drawing landmarks in UI)."""
    return _holistic


def get_last_mp_results():
    """
    Return the most recent MediaPipe Holistic result populated by the last
    predict_frame() call. Used by the display thread to draw the skeleton
    overlay without running Holistic a second time per frame.

    NOTE: returned object is shared with the inference thread. Read-only use
    from the display thread is safe because MediaPipe allocates fresh result
    objects per process() call.
    """
    return _last_mp_results


def reset_window() -> None:
    """Clear prediction window — call when the user explicitly resets."""
    global _lm_prev
    _pred_window.clear()
    _lm_prev = None


# ── Private helpers ───────────────────────────────────────────────────────────

def _emotion_to_onehot(emotion: str) -> np.ndarray:
    vec = np.zeros(EMOTION_DIM, dtype=np.float32)
    idx = (EMOTION_CLASSES.index(emotion)
           if emotion in EMOTION_CLASSES
           else NEUTRAL_IDX)
    vec[idx] = 1.0
    return vec


def _extract_landmarks(frame: np.ndarray) -> np.ndarray:
    global _last_mp_results
    results = _holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    _last_mp_results = results
    feat = []

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 63)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 63)

    if results.face_landmarks:
        for idx in FACE_IDX:
            lm = results.face_landmarks.landmark[idx]
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 30)

    return np.array(feat, dtype=np.float32)


def _normalize(ff: np.ndarray) -> np.ndarray:
    """
    MUST mirror normalize_frame() from Phase 1 notebook Cell 9 exactly.
    Any drift in origin or scale silently shifts the model's input distribution.
    - Hands: wrist (joint 0) -> origin, scale = max radial distance from wrist
    - Face:  centroid       -> origin, scale = max radial distance from centroid
    """
    raw   = ff.astype(np.float64)
    left  = raw[0:63].reshape(21, 3).copy()
    right = raw[63:126].reshape(21, 3).copy()
    face  = raw[126:].reshape(-1, 3).copy()

    if left.any():
        left -= left[0]
        s = np.max(np.linalg.norm(left, axis=1))
        if s > 0:
            left /= s

    if right.any():
        right -= right[0]
        s = np.max(np.linalg.norm(right, axis=1))
        if s > 0:
            right /= s

    if face.any():
        face -= face.mean(axis=0)
        s = np.max(np.linalg.norm(face, axis=1))
        if s > 0:
            face /= s

    return np.concatenate(
        [left.flatten(), right.flatten(), face.flatten()]
    ).astype(np.float32)


def _activation_gate(lm_current: np.ndarray) -> bool:
    global _lm_prev
    if _lm_prev is None:
        _lm_prev = lm_current.copy()
        return False
    # Only measure hand velocity (0:126); face landmarks move during
    # blinking/talking and would cause false activations.
    hands_cur  = lm_current[:126]
    hands_prev = _lm_prev[:126]
    velocity = float(np.linalg.norm(hands_cur - hands_prev))
    _lm_prev = lm_current.copy()
    return velocity >= VELOCITY_THRESHOLD



# Uncomment the block below and comment out load_model() call in demo.py
# to run the UI without the AI model during parallel development.
#
# def predict_frame(frame: np.ndarray) -> tuple:
#     """Stub — returns hardcoded output for UI development."""
#     import time, math
#     labels = ["HELLO", "THANKS", "YES", "NO", "HELP"]
#     label  = labels[int(time.time()) % len(labels)]
#     conf   = 0.7 + 0.25 * abs(math.sin(time.time()))
#     return (label, conf, "happy")
