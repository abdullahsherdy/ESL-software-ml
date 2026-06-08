"""
src/inference.py
================
predict_frame() interface contract

Interface contract (must never change shape without notifying Software team):
    predict_frame(
        frame: np.ndarray,
        cached_emotion: str | None = None,
        *,
        raw: bool = False,
    ) -> tuple[str, float, str]
    Returns: (label, confidence, emotion)

Software team: during Phase 2 setup, replace with the stub at the bottom of
this file until the real model is in place.
"""

import numpy as np
import cv2
import threading
from collections import deque

from src.paths import artifacts_dir, label2idx_path, model_path

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

def _default_model_path() -> str:
    return str(model_path("model_v2.keras"))


def _default_label_path() -> str:
    return str(label2idx_path())

VELOCITY_THRESHOLD = 0.02   # tune empirically; increase if too many false activations
WINDOW_SIZE        = 5
MIN_VOTES          = 3
MIN_CONFIDENCE     = 0.60   # checked against top-label confs only (not all confs)
COMMIT_COOLDOWN    = 8      # compute-frames to wait after each commit (prevents same-sign re-trigger)
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
_pred_window        = deque(maxlen=WINDOW_SIZE)
_cooldown_remaining = 0
_frame_ts_ms        = 0
_frame_ts_lock      = threading.Lock()

_cached_emotion = "neutral"
_emotion_lock   = threading.Lock()

# Most recent MediaPipe Holistic results. Written inside _extract_landmarks()
# (called by predict_frame in the inference thread). Read by the display
# thread via get_last_mp_results() so we do not have to re-run Holistic
# just to draw the skeleton overlay.
_last_mp_results = None

_model_loaded = False

# ── Frame skipping (speed optimization) ────────────────────────────────────────
# Predict every N-th frame; reuse prediction for frames in between.
# Reduces real-time latency by ~3x while maintaining smooth visual feedback.
FRAME_SKIP = 3      # Predict every 3rd frame; skip 2 in between
_frame_counter = 0
_cached_prediction = None


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(
    model_path_arg: str | None = None,
    label_path: str | None = None,
) -> None:
    model_path_arg = model_path_arg or _default_model_path()
    label_path = label_path or _default_label_path()
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

    _model    = tf.keras.models.load_model(model_path_arg)
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
            f"but {model_path_arg} has input_shape={_model.input_shape}. "
            f"Place model_v2.keras in {artifacts_dir()} (MLP + emotion)."
        )

    holistic_model = str(artifacts_dir() / "holistic_landmarker.task")
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=holistic_model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    _holistic = mp.tasks.vision.HolisticLandmarker.create_from_options(options)

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


def _resolve_emotion(cached_emotion: str | None) -> str:
    """Explicit cached_emotion (sprint/web) overrides DeepFace cache."""
    if cached_emotion is not None:
        return cached_emotion
    with _emotion_lock:
        return _cached_emotion


def predict_frame(
    frame: np.ndarray,
    cached_emotion: str | None = None,
    *,
    raw: bool = False,
    disable_frame_skip: bool = False,
) -> tuple:
    """
    Main interface contract.

    Args:
        frame: BGR numpy array (from cv2.VideoCapture or WebSocket decode)
        cached_emotion: If set (e.g. "neutral" for sprint/web), use for one-hot
            concat instead of the DeepFace thread cache. None = full mode.
        raw: Sprint mode — per-frame class label, no internal sliding window.
            Uses "__no_hands__" when no hands (instruction.md contract).
        disable_frame_skip: Force full inference even if skipping enabled
            (use for evaluation, testing).

    Returns:
        (label: str, confidence: float, emotion: str)

    Full mode label states:
        "No hand detected", "Ready", "Detecting", or committed "<SIGN_LABEL>"
    Raw mode label states:
        "__no_hands__" or "<SIGN_LABEL>" with softmax confidence

    Frame Skipping:
        By default, predict_frame() computes inference every FRAME_SKIP frames
        and reuses the previous prediction in between. This reduces real-time
        latency by ~3x without perceptible accuracy loss. Pass
        disable_frame_skip=True to disable (for evaluation/testing).
    """
    global _frame_counter, _cached_prediction
    if not _model_loaded:
        raise RuntimeError("Call load_model() before predict_frame()")

    emotion_str = _resolve_emotion(cached_emotion)

    # Frame skipping: reuse prediction for FRAME_SKIP-1 frames
    _frame_counter += 1
    if not disable_frame_skip and FRAME_SKIP > 1 and _frame_counter % FRAME_SKIP != 0:
        if _cached_prediction is not None:
            # Return cached prediction from previous compute frame
            return _cached_prediction
        # First call: fall through to compute baseline prediction

    small = cv2.resize(frame, (320, 240))
    raw_lm = _extract_landmarks(small)

    if raw_lm[:126].sum() == 0:
        no_hand = "__no_hands__" if raw else "No hand detected"
        result = (no_hand, 0.0, emotion_str)
        _cached_prediction = result
        return result

    if raw:
        norm_lm = _normalize(raw_lm)
        features = np.concatenate(
            [norm_lm, _emotion_to_onehot(emotion_str)]
        ).reshape(1, -1)
        probs = _model(features, training=False)[0].numpy()
        class_idx = int(np.argmax(probs))
        result = (_idx2label[class_idx], float(probs[class_idx]), emotion_str)
        _cached_prediction = result
        return result

    moving = _activation_gate(raw_lm)
    if not moving:
        result = ("Ready", 0.0, emotion_str)
        _cached_prediction = result
        return result

    # Post-commit cooldown: don't accumulate until COMMIT_COOLDOWN active frames
    # have passed. Prevents the same sign immediately re-triggering after commit.
    global _cooldown_remaining
    if _cooldown_remaining > 0:
        _cooldown_remaining -= 1
        result = ("Detecting", 0.0, emotion_str)
        _cached_prediction = result
        return result

    norm_lm = _normalize(raw_lm)
    features = np.concatenate(
        [norm_lm, _emotion_to_onehot(emotion_str)]
    ).reshape(1, -1)

    probs = _model(features, training=False)[0].numpy()
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    label = _idx2label[class_idx]

    _pred_window.append((label, confidence))
    if len(_pred_window) == WINDOW_SIZE:
        labels = [p[0] for p in _pred_window]
        top = max(set(labels), key=labels.count)
        # Use only the top-label's confidences for the threshold check.
        # Using all confs (old bug) caused noise frames with low confidence to
        # drag the mean below 0.60, preventing commit even when 3+ votes agreed.
        top_confs = [c for lbl, c in _pred_window if lbl == top]
        if labels.count(top) >= MIN_VOTES and np.mean(top_confs) >= MIN_CONFIDENCE:
            mod = EMOTION_CONFLICTS.get((top.lower(), emotion_str.lower()), 1.0)
            _pred_window.clear()
            _cooldown_remaining = COMMIT_COOLDOWN
            result = (top, float(np.mean(top_confs)) * mod, emotion_str)
            _cached_prediction = result
            return result

    result = ("Detecting", float(confidence), emotion_str)
    _cached_prediction = result
    return result


def get_raw_landmarks(frame: np.ndarray) -> np.ndarray | None:
    """
    Extract the 156-dim raw landmark vector for a BGR frame.
    Returns None if the model is not loaded. Hands-only velocity uses [:126].
    """
    if not _model_loaded:
        raise RuntimeError("Call load_model() before get_raw_landmarks()")
    small = cv2.resize(frame, (320, 240))
    return _extract_landmarks(small)


def reset_activation_gate() -> None:
    """Reset velocity gate state (sprint demo between signs)."""
    global _lm_prev
    _lm_prev = None


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
    global _frame_counter, _cooldown_remaining
    _pred_window.clear()
    reset_activation_gate()
    _frame_counter = 0
    _cooldown_remaining = 0


def set_frame_skip(skip_rate: int) -> None:
    """
    Configure frame skipping rate for real-time performance.

    Args:
        skip_rate: Predict every N-th frame.
                   1 = no skipping (full accuracy, slow)
                   3 = predict every 3rd frame (3x faster, minimal accuracy loss)
                   5 = predict every 5th frame (5x faster, some accuracy loss)

    Example: set_frame_skip(3)  # Predict every 3rd frame for mobile
    """
    global FRAME_SKIP
    if skip_rate < 1:
        raise ValueError("skip_rate must be >= 1")
    FRAME_SKIP = skip_rate
    print(f"[inference] Frame skip rate set to: every {skip_rate} frames")


# ── Private helpers ───────────────────────────────────────────────────────────

def _emotion_to_onehot(emotion: str) -> np.ndarray:
    vec = np.zeros(EMOTION_DIM, dtype=np.float32)
    idx = (EMOTION_CLASSES.index(emotion)
           if emotion in EMOTION_CLASSES
           else NEUTRAL_IDX)
    vec[idx] = 1.0
    return vec


def _extract_landmarks(frame: np.ndarray) -> np.ndarray:
    global _last_mp_results, _frame_ts_ms
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
    with _frame_ts_lock:
        _frame_ts_ms += 1
        ts = _frame_ts_ms
    results = _holistic.detect_for_video(mp_image, ts)
    _last_mp_results = results
    feat = []

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 63)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 63)

    if results.face_landmarks:
        for idx in FACE_IDX:
            lm = results.face_landmarks[idx]
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
