"""
scripts/export_tflite.py
========================
Converts artifacts/model_v2.keras to artifacts/model_v2.tflite for slower
CPU-only demo machines (plan §2.6 / §0.3 HIGH).

Usage:
    python scripts/export_tflite.py
    python scripts/export_tflite.py --validate        # + check drift vs Keras
    python scripts/export_tflite.py --source artifacts/model_v3.keras \
                                    --dest   artifacts/model_v3.tflite
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def convert(source: str, dest: str) -> None:
    import tensorflow as tf

    print(f"[tflite] Loading {source}…")
    model = tf.keras.models.load_model(source)
    print(f"[tflite] Input shape: {model.input_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Default optimisations keep float32 weights; enabling DEFAULT applies
    # hybrid quantisation which is usually safe for small MLPs. Off by default
    # here to maximise numerical parity with the Keras model.
    tflite_bytes = converter.convert()

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        f.write(tflite_bytes)
    size_kb = os.path.getsize(dest) / 1024.0
    print(f"[tflite] Saved {dest}  ({size_kb:.1f} KB)")


def validate(source: str, dest: str, n_samples: int = 20) -> None:
    """
    Run the same inputs through Keras and TFLite and print max |diff| and
    mean inference latency for both. A healthy MLP conversion should show
    max |diff| well below 1e-3.
    """
    import tensorflow as tf

    keras_model = tf.keras.models.load_model(source)
    interpreter = tf.lite.Interpreter(model_path=dest)
    interpreter.allocate_tensors()

    in_details  = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    # Prefer real val data if it's there; else random.
    val_path = "artifacts/X_mlp_val.npy"
    if keras_model.input_shape[1:] == (163,) and os.path.exists(val_path):
        X = np.load(val_path).astype(np.float32)
        X = X[:n_samples]
    else:
        shape = (n_samples,) + tuple(d or 1 for d in keras_model.input_shape[1:])
        X = np.random.randn(*shape).astype(np.float32)
    print(f"[tflite] Validating on {len(X)} samples of shape {X.shape[1:]}")

    # Keras pass
    k_t0 = time.perf_counter()
    keras_out = keras_model.predict(X, verbose=0)
    k_ms = (time.perf_counter() - k_t0) * 1000 / len(X)

    # TFLite pass (single-sample — TFLite interpreters are typically not batched)
    tflite_out = np.zeros_like(keras_out)
    t_t0 = time.perf_counter()
    for i in range(len(X)):
        sample = X[i:i + 1].astype(in_details["dtype"])
        interpreter.set_tensor(in_details["index"], sample)
        interpreter.invoke()
        tflite_out[i] = interpreter.get_tensor(out_details["index"])[0]
    t_ms = (time.perf_counter() - t_t0) * 1000 / len(X)

    max_abs   = float(np.max(np.abs(keras_out - tflite_out)))
    mean_abs  = float(np.mean(np.abs(keras_out - tflite_out)))
    k_top1    = np.argmax(keras_out,  axis=1)
    t_top1    = np.argmax(tflite_out, axis=1)
    agree     = float(np.mean(k_top1 == t_top1))

    print("\n[tflite] ── Validation report ──────────────────────────────")
    print(f"  max |Keras - TFLite|      : {max_abs:.6e}")
    print(f"  mean |Keras - TFLite|     : {mean_abs:.6e}")
    print(f"  top-1 agreement           : {agree * 100:.1f}%")
    print(f"  Keras  latency / sample   : {k_ms:.2f} ms")
    print(f"  TFLite latency / sample   : {t_ms:.2f} ms")
    if max_abs > 1e-2:
        print("  WARN: drift exceeds 1e-2 — conversion may be lossy.")


def main() -> int:
    p = argparse.ArgumentParser(description="Convert Keras model to TFLite.")
    p.add_argument("--source",   default="artifacts/model_v2.keras")
    p.add_argument("--dest",     default="artifacts/model_v2.tflite")
    p.add_argument("--validate", action="store_true",
                   help="After conversion, run 20 samples through both and report drift.")
    args = p.parse_args()

    if not os.path.isfile(args.source):
        print(f"[tflite] ERROR: {args.source} not found.", file=sys.stderr)
        return 1

    convert(args.source, args.dest)
    if args.validate:
        validate(args.source, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
