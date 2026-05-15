"""
src/evaluate.py
===============

Produces all Phase 3 evaluation outputs:
    - Ablation table (CSV + printed)
    - Confusion matrix heatmap (results/confusion_matrix.png)
    - Per-class classification report
    - Top-3 confused pair visualisation (results/failure_analysis.png)

Usage:
    python -m src.evaluate
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

os.makedirs("results", exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_arrays():
    """
    Load validation arrays and label map.

    The Phase 1 notebook already bakes the neutral-emotion one-hot into the
    saved arrays, so there is only one MLP val array (163-dim) and one LSTM
    val array ((30, 163)). All three model variants use the same 163-dim
    input — model_v1 is trained with X_v1 that also has NEUTRAL_EMO concat.
    """
    import json
    with open("artifacts/label2idx.json", "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(idx2label))]

    X_mlp_val  = np.load("artifacts/X_mlp_val.npy")
    X_lstm_val = np.load("artifacts/X_lstm_val.npy")
    y_val      = np.load("artifacts/y_val.npy")

    return class_names, X_mlp_val, X_lstm_val, y_val


def time_model(model, X, n=100):
    """Measure single-sample inference latency in ms."""
    sample = X[:1]
    times  = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ─── Ablation table ───────────────────────────────────────────────────────────

def run_ablation(class_names, X_mlp_val, X_lstm_val, y_val):
    """
    Evaluate all model variants and return a DataFrame.
    Columns: Model, Val Acc, Macro F1, Latency mean (ms), Latency std (ms)
    """
    import tensorflow as tf

    variants = [
        ("v1: Baseline MLP (no aug)",    "artifacts/model_v1.keras", X_mlp_val),
        ("v2: Aug MLP + Emotion",        "artifacts/model_v2.keras", X_mlp_val),
        ("v3: Aug LSTM + Emotion",       "artifacts/model_v3.keras", X_lstm_val),
    ]

    rows = []
    for name, path, X in variants:
        if not os.path.exists(path):
            print(f"  SKIP {name} — {path} not found")
            continue
        model   = tf.keras.models.load_model(path)
        _, acc  = model.evaluate(X, y_val, verbose=0)
        y_pred  = np.argmax(model.predict(X, verbose=0), axis=1)
        f1      = f1_score(y_val, y_pred, average="macro", zero_division=0)
        lat_m, lat_s = time_model(model, X)
        rows.append({
            "Model":              name,
            "Val Acc":            round(acc,  4),
            "Macro F1":           round(f1,   4),
            "Latency mean (ms)":  round(lat_m, 2),
            "Latency std (ms)":   round(lat_s, 2),
        })
        print(f"  {name}: acc={acc:.4f}  F1={f1:.4f}  lat={lat_m:.1f}±{lat_s:.1f} ms")

    df = pd.DataFrame(rows)
    df.to_csv("results/ablation_table.csv", index=False)
    print("\nAblation table saved → results/ablation_table.csv")
    return df


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_val, y_val, class_names,
                          title="Confusion Matrix — model_v2",
                          save_path="results/confusion_matrix.png"):
    """Generate and save a seaborn confusion matrix heatmap."""
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm     = confusion_matrix(y_val, y_pred)

    fig_size = max(12, len(class_names) * 0.5)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm,
        annot=True if len(class_names) <= 20 else False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
        linecolor="lightgrey",
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True",      fontsize=12, labelpad=10)
    ax.set_title(title,        fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {save_path}")

    # Return top confused pairs for failure analysis
    np.fill_diagonal(cm, 0)   # zero diagonal before finding off-diagonal maxima
    confused_pairs = []
    flat_sorted = np.argsort(cm.ravel())[::-1]
    for idx in flat_sorted[:6]:
        row, col = divmod(idx, len(class_names))
        if row != col and cm[row, col] > 0:
            confused_pairs.append((class_names[row], class_names[col], cm[row, col]))
        if len(confused_pairs) == 3:
            break
    return confused_pairs, y_pred


# ─── Per-class report ─────────────────────────────────────────────────────────

def print_per_class_report(y_true, y_pred, class_names):
    """Print sklearn classification report and save to results/."""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print("\n── Per-class Classification Report ───────────────────────────────")
    print(report)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)
    print("Report saved → results/classification_report.txt")


# ─── Failure analysis ─────────────────────────────────────────────────────────

def plot_failure_analysis(confused_pairs, X_val, y_val, class_names,
                          save_path="results/failure_analysis.png"):
    """
    For the top 3 confused pairs, plot the landmark sequences side-by-side.
    This visualises *why* the model confuses them — the viva's strongest moment.
    """
    if not confused_pairs:
        print("No confused pairs to visualise.")
        return

    n = len(confused_pairs)
    fig = plt.figure(figsize=(18, 5 * n))
    fig.suptitle("Failure Analysis — Top Confused Sign Pairs",
                 fontsize=15, fontweight="bold", y=1.01)

    for pair_idx, (true_lbl, pred_lbl, count) in enumerate(confused_pairs):
        true_idx = class_names.index(true_lbl)
        pred_idx = class_names.index(pred_lbl)

        # Find a sample of each class from validation set
        true_samples = X_val[y_val == true_idx]
        pred_samples = X_val[y_val == pred_idx]

        if len(true_samples) == 0 or len(pred_samples) == 0:
            continue

        # Use the first sample's landmark sequence
        # For MLP: shape (163,) → reshape to (21+21+10, 3) meaningful joints
        # We plot the x/y of all 52 landmarks as a scatter
        def _landmark_scatter(ax, feat, title):
            if feat.ndim == 1:
                # MLP feature: 163 = 156 landmarks + 7 emotion (drop emotion tail)
                pts = feat[:156].reshape(-1, 3)[:, :2]   # x, y only
            else:
                # LSTM feature: (T, 163), use middle frame
                mid = feat.shape[0] // 2
                pts = feat[mid, :156].reshape(-1, 3)[:, :2]

            colours = (["#2ecc71"] * 21 +   # left hand
                       ["#3498db"] * 21 +   # right hand
                       ["#e74c3c"] * 10)    # face
            ax.scatter(pts[:, 0], -pts[:, 1],
                       c=colours[:len(pts)], s=40, zorder=3)
            # Connect hand landmarks sequentially (rough skeleton)
            for start, end in [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20),
                                (0, 5), (0, 17)]:
                if start < len(pts) and end < len(pts):
                    ax.plot([pts[start, 0], pts[end, 0]],
                            [-pts[start, 1], -pts[end, 1]],
                            "grey", linewidth=0.8, zorder=2)
                    # right hand offset
                    rs, re = start + 21, end + 21
                    if rs < len(pts) and re < len(pts):
                        ax.plot([pts[rs, 0], pts[re, 0]],
                                [-pts[rs, 1], -pts[re, 1]],
                                "lightgrey", linewidth=0.8, zorder=2)

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#1a1a2e")

        ax1 = fig.add_subplot(n, 3, pair_idx * 3 + 1)
        ax2 = fig.add_subplot(n, 3, pair_idx * 3 + 2)
        ax3 = fig.add_subplot(n, 3, pair_idx * 3 + 3)

        _landmark_scatter(ax1, true_samples[0],
                          f"TRUE: {true_lbl}\n(green=L  blue=R  red=face)")
        _landmark_scatter(ax2, pred_samples[0],
                          f"CONFUSED WITH: {pred_lbl}\n({count} misclassifications)")

        # Overlay both on ax3 to show similarity
        if true_samples[0].ndim == 1:
            pts_t = true_samples[0][:156].reshape(-1, 3)[:, :2]
            pts_p = pred_samples[0][:156].reshape(-1, 3)[:, :2]
        else:
            mid = true_samples[0].shape[0] // 2
            pts_t = true_samples[0][mid, :156].reshape(-1, 3)[:, :2]
            pts_p = pred_samples[0][mid, :156].reshape(-1, 3)[:, :2]

        ax3.scatter(pts_t[:, 0], -pts_t[:, 1], c="#2ecc71", s=30,
                    alpha=0.8, label=true_lbl)
        ax3.scatter(pts_p[:, 0], -pts_p[:, 1], c="#e74c3c", s=30,
                    alpha=0.6, label=pred_lbl, marker="x")
        ax3.set_title("Overlay — why they're confused", fontsize=11)
        ax3.legend(fontsize=8, loc="upper right")
        ax3.set_aspect("equal")
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_facecolor("#1a1a2e")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Failure analysis saved → {save_path}")


# ─── Stability / memory log ───────────────────────────────────────────────────

def log_memory_usage(duration_sec=60, interval_sec=5):
    """
    Log process RSS memory every interval_sec for duration_sec.
    Used in Task 3.2 stability testing.
    Saves results/memory_log.csv.
    """
    import tracemalloc
    import csv

    tracemalloc.start()
    rows = []
    start = time.time()
    print(f"[stability] Memory logging for {duration_sec}s every {interval_sec}s…")
    while time.time() - start < duration_sec:
        current, peak = tracemalloc.get_traced_memory()
        rows.append({
            "elapsed_s":  round(time.time() - start, 1),
            "current_MB": round(current / 1024 / 1024, 2),
            "peak_MB":    round(peak    / 1024 / 1024, 2),
        })
        time.sleep(interval_sec)

    tracemalloc.stop()
    df = pd.DataFrame(rows)
    df.to_csv("results/memory_log.csv", index=False)
    print(df.to_string(index=False))
    print("Memory log saved → results/memory_log.csv")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import tensorflow as tf

    print("=" * 65)
    print("ESL Phase 3 — Evaluation")
    print("=" * 65)

    class_names, X_mlp_val, X_lstm_val, y_val = load_arrays()
    print(f"Val set: {len(y_val)} samples  |  {len(class_names)} classes\n")

    # 1. Ablation table
    print("── Ablation Table ────────────────────────────────────────────")
    df_ablation = run_ablation(class_names, X_mlp_val, X_lstm_val, y_val)
    print("\n" + df_ablation.to_string(index=False))

    # 2. Confusion matrix (use primary model: v2)
    if os.path.exists("artifacts/model_v2.keras"):
        print("\n── Confusion Matrix ──────────────────────────────────────────")
        model_v2 = tf.keras.models.load_model("artifacts/model_v2.keras")
        confused_pairs, y_pred = plot_confusion_matrix(
            model_v2, X_mlp_val, y_val, class_names,
        )
        print(f"Top confused pairs: {confused_pairs}")

        # 3. Per-class report
        print_per_class_report(y_val, y_pred, class_names)

        # 4. Failure analysis
        print("\n── Failure Analysis ──────────────────────────────────────────")
        plot_failure_analysis(confused_pairs, X_mlp_val, y_val, class_names)

    print("\n" + "=" * 65)
    print("Evaluation complete. All outputs in results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
