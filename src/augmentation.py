"""
src/augmentation.py
===================

All data augmentation logic for landmark sequences.
Imported by the Phase 1 notebook and can be run standalone.

Usage:
    from src.augmentation import augment_sequence, augment_dataset
"""

import numpy as np
from scipy.interpolate import interp1d


# ── Individual augmentation functions ─────────────────────────────────────────
def aug_hflip(seq: np.ndarray) -> np.ndarray:
    """
    Horizontal flip: negate all x-coordinates and swap left/right hand blocks.
    On normalised (wrist-centred) data, flip = negate x + swap hands.
    Must match flip_horizontal() in the Phase 1 notebook exactly.
    """
    out = seq.copy()
    # Negate all x-coordinates (every 3rd value starting at 0)
    out[..., 0::3] = -out[..., 0::3]
    # Swap left hand (0:63) and right hand (63:126) blocks
    left_orig = seq[..., 0:63].copy()
    out[..., 0:63]   = seq[..., 63:126]
    out[..., 63:126] = left_orig
    return out


def aug_time_jitter(seq: np.ndarray, factor: float) -> np.ndarray:
    """
    Temporal speed jitter: resample to factor×length, trim/pad to original length.
    factor < 1.0 → sign is performed faster (e.g. 0.8)
    factor > 1.0 → sign is performed slower (e.g. 1.2)
    """
    T = len(seq)
    T_new = max(2, int(T * factor))

    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, T_new)

    f   = interp1d(old_t, seq, axis=0, kind="linear")
    out = f(new_t)

    if T_new >= T:
        return out[:T].astype(np.float32)
    else:
        pad = np.zeros((T - T_new, seq.shape[1]), dtype=np.float32)
        return np.vstack([out, pad]).astype(np.float32)


def aug_gaussian_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to all landmark coordinates.
    Simulates natural hand tremor and imperfect MediaPipe detection.
    Mean=0 preserves the sign shape on average.
    """
    return (seq + np.random.normal(0, sigma, seq.shape)).astype(np.float32)


def aug_landmark_dropout(seq: np.ndarray,
                         drop_prob: float = 0.1,
                         num_joints: int = 2) -> np.ndarray:
    """
    Zero-out 1–2 random non-wrist hand joints per frame with probability drop_prob.
    The wrist anchor point (joint 0 for each hand) is NEVER dropped.
    Simulates partial occlusion.

    Feature layout assumed: [left_hand(63), right_hand(63), face(30)]
    Left hand joints 1-20 start at offsets 3, 6, ... 60
    Right hand joints 1-20 start at offsets 66, 69, ... 123
    """
    out = seq.copy()
    T   = len(out)

    # Offsets of non-wrist joints (skip wrist = joint 0 = offsets 0 and 63)
    hand_joint_offsets = (
        [i * 3 for i in range(1, 21)] +          # left hand joints 1-20
        [63 + i * 3 for i in range(1, 21)]        # right hand joints 1-20
    )

    for t in range(T):
        if np.random.rand() < drop_prob:
            chosen = np.random.choice(hand_joint_offsets, size=num_joints, replace=False)
            for off in chosen:
                out[t, off:off + 3] = 0.0

    return out


def aug_rotation_2d(seq: np.ndarray, max_angle_deg: float = 10.0) -> np.ndarray:
    """
    Apply a small random 2D rotation to all hand x/y coordinates.
    Rotation is around the wrist centre of each hand.
    Simulates slight variation in signing angle.
    """
    angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    out = seq.copy()

    def _rotate_hand(hand_slice, anchor_offset):
        """Rotate 21 joints around joint 0 (wrist)."""
        pts = hand_slice.reshape(21, 3)
        pivot = pts[0, :2].copy()
        xy    = pts[:, :2] - pivot
        rot   = np.stack([
            cos_a * xy[:, 0] - sin_a * xy[:, 1],
            sin_a * xy[:, 0] + cos_a * xy[:, 1],
        ], axis=1)
        pts[:, :2] = rot + pivot
        return pts.flatten()

    out[:, 0:63]  = np.array([_rotate_hand(out[t, 0:63],  0) for t in range(len(out))])
    out[:, 63:126] = np.array([_rotate_hand(out[t, 63:126], 0) for t in range(len(out))])

    return out.astype(np.float32)


# ── Augment a single sequence (all strategies) ────────────────────────────────

def augment_sequence(seq: np.ndarray, include_rotation: bool = False) -> list:
    """
    Apply all augmentation strategies to a single (T, D) sequence.
    Returns list of augmented sequences (NOT including the original).

    Args:
        seq:              shape (T, D) — normalised landmark sequence
        include_rotation: add 2D rotation variant (optional, off by default)

    Returns:
        list of (T, D) arrays
    """
    variants = [
        aug_hflip(seq),
        aug_time_jitter(seq, 0.80),
        aug_time_jitter(seq, 1.20),
        aug_gaussian_noise(seq, sigma=0.005),
        aug_landmark_dropout(seq, drop_prob=0.1, num_joints=2),
    ]
    if include_rotation:
        variants.append(aug_rotation_2d(seq))
    return variants


# ── Augment an entire dataset ──────────────────────────────────────────────────
def augment_dataset(X: np.ndarray, y: np.ndarray,
                    include_rotation: bool = False,
                    seed: int = 42) -> tuple:
    """
    Augment the full training dataset.

    Args:
        X:    (N, T, D) array of LSTM sequences
        y:    (N,) label array
        seed: random seed for reproducibility

    Returns:
        X_aug, y_aug — includes originals + all augmented variants, shuffled
    """
    np.random.seed(seed)

    X_all = [X]
    y_all = [y]

    for seq, lbl in zip(X, y):
        for aug_seq in augment_sequence(seq, include_rotation=include_rotation):
            X_all.append(aug_seq[np.newaxis])
            y_all.append([lbl])

    X_aug = np.concatenate(X_all, axis=0).astype(np.float32)
    y_aug = np.concatenate(y_all, axis=0)

    # Shuffle
    perm  = np.random.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# ── Build augmented MLP features from LSTM sequences ──────────────────────────
def lstm_to_mlp_features(X_lstm: np.ndarray) -> np.ndarray:
    """
    Convert (N, T, D) LSTM sequences to (N, D) MLP features
    by averaging the 5 centre frames.
    """
    N, T, D = X_lstm.shape
    mid     = T // 2
    indices = list(range(max(0, mid - 2), min(T, mid + 3)))
    return X_lstm[:, indices, :].mean(axis=1).astype(np.float32)


# ── Emotion feature concat ────────────────────────────────────────────────────

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NEUTRAL_VEC     = np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32)


def add_neutral_emotion(X: np.ndarray) -> np.ndarray:
    """
    Concatenate the neutral emotion one-hot to every sample.
    For MLP: (N, D) → (N, D+7)
    For LSTM: (N, T, D) → (N, T, D+7)
    """
    if X.ndim == 2:
        emo_mat = np.tile(NEUTRAL_VEC, (len(X), 1))
        return np.hstack([X, emo_mat]).astype(np.float32)
    elif X.ndim == 3:
        N, T, _  = X.shape
        emo_tens = np.tile(NEUTRAL_VEC, (N, T, 1))
        return np.concatenate([X, emo_tens], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")
