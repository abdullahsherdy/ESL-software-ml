"""
scripts/export_eval_arrays.py
=============================
Regenerates the Phase 3 evaluation arrays from the per-video landmark .npy
files on disk, so `python -m src.evaluate` can run without a GPU or a fresh
Phase 1 notebook run.

Outputs (overwrites artifacts/*):
    X_mlp_train.npy    (N_train, 163)
    X_mlp_val.npy      (N_val,   163)
    X_lstm_train.npy   (N_train, 30, 163)
    X_lstm_val.npy     (N_val,   30, 163)
    y_train.npy        (N_train,)
    y_val.npy          (N_val,)

Reconstruction logic mirrors Phase 1 notebook cells 5 (split), 9 (already
applied to .npy files on disk), 13 (MLP/LSTM feature build) with:
    RANDOM_SEED = 42
    TRAIN_SPLIT = 0.8
    N_FRAMES    = 30
    EMOTION     = neutral one-hot (index 4 of 7)

Usage:
    python scripts/export_eval_arrays.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

# ── Constants (must match Phase 1 notebook) ──────────────────────────────────
RANDOM_SEED   = 42
TRAIN_SPLIT   = 0.8
N_FRAMES      = 30
FEATURE_DIM   = 156
EMOTION_DIM   = 7
NEUTRAL_IDX   = 4      # "neutral" position in EMOTION_CLASSES

ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKS_DIR   = os.path.join(ROOT, "data", "landmarks")
AUG_DIR         = os.path.join(ROOT, "data", "augmented_landmarks")
ARTIFACTS_DIR   = os.path.join(ROOT, "artifacts")
LABEL2IDX_PATH  = os.path.join(ARTIFACTS_DIR, "label2idx.json")

AUG_SUFFIXES    = ("_aug_flip", "_aug_slow", "_aug_fast", "_aug_noise")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_label_map() -> dict:
    with open(LABEL2IDX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_class(stem: str, classes: list[str]) -> str | None:
    """
    Filenames look like  '{class_name}_{video_id}'  where class_name may itself
    contain spaces or underscores. Resolve by matching the longest known class
    name that the filename starts with + '_'.
    """
    for cls in sorted(classes, key=len, reverse=True):
        prefix = cls + "_"
        if stem.startswith(prefix):
            return cls
    return None


def strip_aug_suffix(stem: str) -> tuple[str, bool]:
    """Return (original_stem, is_augmented). Augmented stems end with _aug_*."""
    for suf in AUG_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)], True
    return stem, False


def pad_or_trim_center(seq: np.ndarray, N: int) -> np.ndarray:
    """Match Phase 1 notebook CELL 13 exactly."""
    L = len(seq)
    if L >= N:
        start = (L - N) // 2
        return seq[start:start + N].astype(np.float32)
    before = (N - L) // 2
    after  = N - L - before
    return np.vstack([
        np.zeros((before, seq.shape[1]), np.float32),
        seq,
        np.zeros((after,  seq.shape[1]), np.float32),
    ])


def build_split(originals: dict[str, list[str]]) -> tuple[dict, dict]:
    """
    Video-level 80/20 split per class — exactly mirrors Phase 1 notebook
    CELL 5 (random.seed(RANDOM_SEED) then random.shuffle per class).
    """
    random.seed(RANDOM_SEED)
    train_split: dict[str, list[str]] = defaultdict(list)
    val_split:   dict[str, list[str]] = defaultdict(list)

    for cls in sorted(originals.keys()):
        videos = sorted(originals[cls])
        random.shuffle(videos)
        n = len(videos)
        idx = max(1, min(n - 1, math.floor(n * TRAIN_SPLIT))) if n > 1 else n
        train_split[cls] = videos[:idx]
        val_split[cls]   = videos[idx:]
    return train_split, val_split


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not os.path.isdir(LANDMARKS_DIR):
        print(f"[export] ERROR: {LANDMARKS_DIR} not found.", file=sys.stderr)
        return 1
    if not os.path.isfile(LABEL2IDX_PATH):
        print(f"[export] ERROR: {LABEL2IDX_PATH} not found.", file=sys.stderr)
        return 1

    label2idx = load_label_map()
    classes   = list(label2idx.keys())
    print(f"[export] {len(classes)} classes in label2idx.json")

    # ── 1. Discover originals grouped by class ───────────────────────────────
    originals: dict[str, list[str]] = defaultdict(list)
    for fname in os.listdir(LANDMARKS_DIR):
        if not fname.endswith(".npy"):
            continue
        stem = fname[:-4]
        cls  = parse_class(stem, classes)
        if cls is None:
            print(f"[export] WARN: skip unknown class for {fname}")
            continue
        originals[cls].append(stem)

    n_total = sum(len(v) for v in originals.values())
    print(f"[export] Found {n_total} original videos across {len(originals)} classes")

    train_split, val_split = build_split(originals)
    train_stems = {s for lst in train_split.values() for s in lst}
    val_stems   = {s for lst in val_split.values()   for s in lst}
    print(f"[export] Split: {len(train_stems)} train videos / {len(val_stems)} val videos")

    # ── 2. Build MLP + LSTM arrays ──────────────────────────────────────────
    neutral_emo = np.zeros(EMOTION_DIM, dtype=np.float32)
    neutral_emo[NEUTRAL_IDX] = 1.0
    emo_tiled   = np.tile(neutral_emo, (N_FRAMES, 1))

    Xml_tr, Xml_va = [], []
    Xls_tr, Xls_va = [], []
    y_tr,   y_va   = [], []

    def _append(split: str, mlp_vec: np.ndarray, lstm_mat: np.ndarray, lbl: int) -> None:
        if split == "train":
            Xml_tr.append(mlp_vec); Xls_tr.append(lstm_mat); y_tr.append(lbl)
        else:
            Xml_va.append(mlp_vec); Xls_va.append(lstm_mat); y_va.append(lbl)

    def _build_features(seq: np.ndarray, lbl: int, split: str) -> None:
        if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM or seq.shape[0] == 0:
            return
        mlp  = np.concatenate([seq.mean(axis=0), neutral_emo]).astype(np.float32)
        lstm = np.concatenate(
            [pad_or_trim_center(seq, N_FRAMES), emo_tiled], axis=1
        ).astype(np.float32)
        _append(split, mlp, lstm, lbl)

    # ── 2a. Originals ────────────────────────────────────────────────────────
    kept = skipped = 0
    for fname in sorted(os.listdir(LANDMARKS_DIR)):
        if not fname.endswith(".npy"):
            continue
        stem = fname[:-4]
        cls  = parse_class(stem, classes)
        if cls is None:
            continue
        split = "train" if stem in train_stems else "val"
        try:
            seq = np.load(os.path.join(LANDMARKS_DIR, fname))
        except Exception as e:
            print(f"[export] WARN: could not load {fname}: {e}")
            skipped += 1
            continue
        _build_features(seq, label2idx[cls], split)
        kept += 1
    print(f"[export] Originals: kept {kept}  skipped {skipped}")

    # ── 2b. Augmented (always train) ────────────────────────────────────────
    if os.path.isdir(AUG_DIR):
        aug_kept = aug_skipped = aug_val_drop = 0
        for fname in sorted(os.listdir(AUG_DIR)):
            if not fname.endswith(".npy"):
                continue
            stem              = fname[:-4]
            orig_stem, is_aug = strip_aug_suffix(stem)
            if not is_aug:
                continue
            cls = parse_class(orig_stem, classes)
            if cls is None:
                aug_skipped += 1
                continue
            # Guard: augmented files are only produced for train-split videos;
            # if somehow an aug of a val video exists, drop it (data leakage).
            if orig_stem in val_stems:
                aug_val_drop += 1
                continue
            try:
                seq = np.load(os.path.join(AUG_DIR, fname))
            except Exception as e:
                print(f"[export] WARN: could not load {fname}: {e}")
                aug_skipped += 1
                continue
            _build_features(seq, label2idx[cls], "train")
            aug_kept += 1
        print(f"[export] Augmented: kept {aug_kept}  skipped {aug_skipped}  val-leak-drop {aug_val_drop}")
    else:
        print(f"[export] NOTE: {AUG_DIR} missing — training arrays will not include augmentations.")

    # ── 3. Save ──────────────────────────────────────────────────────────────
    X_mlp_train  = np.asarray(Xml_tr, dtype=np.float32)
    X_mlp_val    = np.asarray(Xml_va, dtype=np.float32)
    X_lstm_train = np.asarray(Xls_tr, dtype=np.float32)
    X_lstm_val   = np.asarray(Xls_va, dtype=np.float32)
    y_train      = np.asarray(y_tr,   dtype=np.int32)
    y_val        = np.asarray(y_va,   dtype=np.int32)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    outputs = [
        ("X_mlp_train.npy",  X_mlp_train),
        ("X_mlp_val.npy",    X_mlp_val),
        ("X_lstm_train.npy", X_lstm_train),
        ("X_lstm_val.npy",   X_lstm_val),
        ("y_train.npy",      y_train),
        ("y_val.npy",        y_val),
    ]
    for name, arr in outputs:
        path = os.path.join(ARTIFACTS_DIR, name)
        np.save(path, arr)
        print(f"[export] Saved {path}  shape={arr.shape}  dtype={arr.dtype}")

    # ── 4. Class balance sanity ─────────────────────────────────────────────
    uniq, cnt = np.unique(y_val, return_counts=True)
    print(f"\n[export] Val class coverage: {len(uniq)}/{len(classes)} classes have at least one sample")
    print(f"[export] Train samples: {len(y_train)}  |  Val samples: {len(y_val)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
