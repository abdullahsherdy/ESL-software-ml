"""
src/landmark_gate.py
====================
Activation gate helpers for sprint mode (demo-layer smoothing).
Uses get_raw_landmarks() from inference — no duplicate MediaPipe logic.
"""

from __future__ import annotations

import numpy as np

from src.inference import VELOCITY_THRESHOLD, get_raw_landmarks

_lm_prev: np.ndarray | None = None


def reset_gate() -> None:
    global _lm_prev
    _lm_prev = None


def activation_gate(feat_126: np.ndarray) -> bool:
    """
    Return True when hand velocity exceeds VELOCITY_THRESHOLD.
    feat_126: first 126 values of the raw landmark array (both hands).
    """
    global _lm_prev
    if _lm_prev is None:
        _lm_prev = feat_126.copy()
        return True
    velocity = float(np.linalg.norm(feat_126 - _lm_prev))
    _lm_prev = feat_126.copy()
    return velocity >= VELOCITY_THRESHOLD


def gate_from_frame(frame: np.ndarray) -> tuple[bool, np.ndarray | None]:
    """
    Extract landmarks and run the activation gate.
    Returns (is_active, raw_landmarks_or_none).
    """
    raw = get_raw_landmarks(frame)
    if raw is None or raw[:126].sum() == 0:
        return False, raw
    return activation_gate(raw[:126]), raw
