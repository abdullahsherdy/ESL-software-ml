"""
demo.py
=======
ESL Real-Time Sign Language Recognition — Desktop Application
Phase 2 deliverable.

Usage:
    python demo.py                      # live webcam
    python demo.py --source video.mp4   # video file (for testing / demo recording)
    python demo.py --source 0           # explicit camera index

Keyboard controls:
    Q — quit
    L — toggle landmark skeleton overlay
    D — toggle debug mode (raw per-frame prediction probabilities)
    R — reset prediction window
"""

import argparse
import queue
import threading
import time
import os
from collections import deque

import cv2
import numpy as np

# Hand skeleton edges (MediaPipe 0.10.30+ removed mp.solutions drawing_utils)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

from src.inference import (
    load_model,
    predict_frame,
    update_emotion_async,
    get_last_mp_results,
    reset_window,
    DEEPFACE_INTERVAL,
)

# ── UI Constants ──────────────────────────────────────────────────────────────
WIN_NAME    = "ESL Sign Language Recognition"
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FPS_WINDOW  = 30          # rolling average over this many frames

# Colours (BGR)
C_GREEN   = (0,   210,  80)
C_YELLOW  = (0,   200, 220)
C_RED     = (50,   50, 230)
C_WHITE   = (255, 255, 255)
C_GREY    = (160, 160, 160)
C_BLACK   = (0,     0,   0)
C_OVERLAY = (20,   20,  20)   # semi-transparent overlay background

# State colours
STATE_COLOURS = {
    "No hand detected": C_GREY,
    "Ready": C_YELLOW,
    "Detecting": C_GREEN,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Thread workers
# ═══════════════════════════════════════════════════════════════════════════════

def capture_thread(source, frame_queue: queue.Queue, stop_event: threading.Event):
    """
    Thread 1 — Capture
    Reads frames from webcam / video at native FPS.
    Drops old frames if queue is full to prevent lag accumulation.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[capture] ERROR: Cannot open source '{source}'")
        stop_event.set()
        return

    print(f"[capture] Opened source: {source}  "
          f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
          f"{cap.get(cv2.CAP_PROP_FPS):.0f} FPS)")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[capture] Stream ended or read error.")
            stop_event.set()
            break

        # Drop old frame if queue is full — prevents ever-growing lag
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

    cap.release()
    print("[capture] Thread stopped.")

def inference_thread(frame_queue: queue.Queue,
                     result_queue: queue.Queue,
                     stop_event: threading.Event):
    """
    Thread 2 — Inference
    Pops frames → MediaPipe landmarks → activation gate → model.predict()
    Pushes (label, confidence, emotion, mediapipe_results) to result_queue.
    """
    frame_count = 0
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        frame_count += 1
        try:
            label, conf, emotion = predict_frame(frame)
            mp_results = get_last_mp_results()
        except Exception as e:
            print(f"[inference] predict_frame error: {e}")
            label, conf, emotion = "Error", 0.0, "neutral"
            mp_results = None

        # Pass the raw frame + MediaPipe results alongside the prediction so
        # the display thread can draw the skeleton overlay without running
        # Holistic a second time per frame.
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        result_queue.put((frame.copy(), label, conf, emotion, frame_count, mp_results))

    print("[inference] Thread stopped.")

def emotion_thread(frame_queue_ref,
                   stop_event: threading.Event,
                   interval: int = DEEPFACE_INTERVAL):
    """
    Thread 3 — Emotion
    Every `interval` calls, runs DeepFace on the latest frame.
    Uses update_emotion_async() which writes to the shared _cached_emotion
    inside inference.py via a threading.Lock.
    """
    call_count = 0
    last_frame = None

    # We spy on the frame queue without consuming from it by keeping our own
    # reference to the most recently seen frame via a simple list[1] hack.
    while not stop_event.is_set():
        time.sleep(0.05)   # poll at 20 Hz

        if frame_queue_ref[0] is None:
            continue

        call_count += 1
        if call_count % interval == 0:
            frame = frame_queue_ref[0]
            try:
                update_emotion_async(frame)
            except Exception as e:
                print(f"[emotion] DeepFace error: {e}")

    print("[emotion] Thread stopped.")

# ═══════════════════════════════════════════════════════════════════════════════
#  Display / overlay helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_rounded_rect(img, x1, y1, x2, y2, colour, radius=8, alpha=0.55):
    """Draw a semi-transparent rounded rectangle overlay."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_status_bar(frame, label: str, conf: float, emotion: str,
                    fps: float, debug_probs: dict | None = None,
                    show_debug: bool = False):
    """Draw the full UI overlay on the frame (in-place)."""
    h, w = frame.shape[:2]

    # ── Bottom status bar background ─────────────────────────────────────────
    bar_h = 90
    draw_rounded_rect(frame, 0, h - bar_h, w, h, C_OVERLAY, radius=0, alpha=0.7)

    # ── State colour and status text ─────────────────────────────────────────
    if label in STATE_COLOURS:
        colour = STATE_COLOURS[label]
        status_text = label
    else:
        colour = C_GREEN
        status_text = f"Sign: {label}"

    # Status dot
    cv2.circle(frame, (18, h - bar_h + 20), 8, colour, -1)

    # Main label
    cv2.putText(frame, status_text,
                (35, h - bar_h + 28),
                FONT, 0.8, colour, 2, cv2.LINE_AA)

    # Confidence bar (only when a real prediction is showing)
    if conf > 0 and label not in STATE_COLOURS:
        bar_x, bar_y = 35, h - bar_h + 42
        bar_w_max    = 250
        bar_w_fill   = int(bar_w_max * min(conf, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 10), C_GREY, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_fill, bar_y + 10), colour, -1)
        cv2.putText(frame, f"{conf * 100:.0f}%",
                    (bar_x + bar_w_max + 8, bar_y + 10),
                    FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)

    # Emotion
    emo_text = f"Emotion: {emotion}"
    cv2.putText(frame, emo_text,
                (35, h - bar_h + 72),
                FONT, 0.55, C_YELLOW, 1, cv2.LINE_AA)

    # FPS counter (top right)
    fps_text = f"{fps:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, FONT, 0.5, 1)
    cv2.putText(frame, fps_text,
                (w - tw - 10, 22),
                FONT, 0.5, C_WHITE, 1, cv2.LINE_AA)

    # Debug panel
    if show_debug and debug_probs:
        _draw_debug_panel(frame, debug_probs)

def _draw_debug_panel(frame, probs: dict):
    """Draw top-5 class probabilities as a small panel (debug mode)."""
    h, w = frame.shape[:2]
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    panel_x, panel_y = w - 220, 35
    draw_rounded_rect(frame, panel_x - 5, panel_y - 5,
                      w - 5, panel_y + len(top5) * 22 + 8,
                      C_OVERLAY, alpha=0.75)
    cv2.putText(frame, "DEBUG — top 5",
                (panel_x, panel_y + 10), FONT, 0.4, C_GREY, 1)
    for i, (lbl, p) in enumerate(top5):
        y = panel_y + 28 + i * 20
        bar = int(100 * p)
        cv2.rectangle(frame, (panel_x, y - 10), (panel_x + bar, y), C_GREEN, -1)
        cv2.putText(frame, f"{lbl[:12]:12s} {p:.2f}",
                    (panel_x, y), FONT, 0.38, C_WHITE, 1)

def _draw_hand_skeleton(frame, landmarks, w: int, h: int) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], C_GREEN, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def draw_landmarks(frame, mediapipe_results):
    """Overlay hand landmarks from HolisticLandmarker (tasks API)."""
    if mediapipe_results is None:
        return
    h, w = frame.shape[:2]
    if mediapipe_results.left_hand_landmarks:
        _draw_hand_skeleton(frame, mediapipe_results.left_hand_landmarks, w, h)
    if mediapipe_results.right_hand_landmarks:
        _draw_hand_skeleton(frame, mediapipe_results.right_hand_landmarks, w, h)

# ═══════════════════════════════════════════════════════════════════════════════
#  Main application
# ═══════════════════════════════════════════════════════════════════════════════

def run(source):
    """Launch the full 3-thread + display pipeline."""
    print("[app] Loading model…")
    load_model()
    print("[app] Model ready. Starting threads…")

    # Queues
    frame_queue  = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    stop_event   = threading.Event()

    # Shared frame reference for emotion thread (list[1] trick — mutable)
    latest_frame_ref = [None]

    # Threads
    t1 = threading.Thread(
        target=capture_thread,
        args=(source, frame_queue, stop_event),
        daemon=True, name="T1-Capture",
    )
    t2 = threading.Thread(
        target=inference_thread,
        args=(frame_queue, result_queue, stop_event),
        daemon=True, name="T2-Inference",
    )
    t3 = threading.Thread(
        target=emotion_thread,
        args=(latest_frame_ref, stop_event),
        daemon=True, name="T3-Emotion",
    )

    t1.start(); t2.start(); t3.start()
    print("[app] All threads started. Press Q to quit, L to toggle landmarks, D for debug.")

    # ── Display state ──────────────────────────────────────────────────────
    show_landmarks  = True
    show_debug      = False
    fps_deque       = deque(maxlen=FPS_WINDOW)
    last_label      = "Waiting…"
    last_conf       = 0.0
    last_emotion    = "neutral"
    t_last          = time.perf_counter()

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 800, 520)

    while not stop_event.is_set():
        try:
            frame, label, conf, emotion, fcount, mp_results = result_queue.get(timeout=0.5)
        except queue.Empty:
            # Nothing new; check for key press and loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # Update latest frame for emotion thread
        latest_frame_ref[0] = frame

        # Update cached values only on meaningful transitions
        if label not in ("Detecting", "Error"):
            last_label   = label
            last_conf    = conf
            last_emotion = emotion

        # FPS
        now = time.perf_counter()
        fps_deque.append(now - t_last)
        t_last = now
        fps = 1.0 / (sum(fps_deque) / len(fps_deque)) if fps_deque else 0.0

        # Optionally draw landmarks. Reuse the MediaPipe results captured by
        # the inference thread instead of running Holistic a second time.
        # Holistic results are in normalised [0,1] coords, so MediaPipe's
        # draw helpers scale them to whatever target frame we pass.
        disp_frame = frame.copy()
        if show_landmarks and mp_results is not None:
            draw_landmarks(disp_frame, mp_results)

        # Draw status bar
        draw_status_bar(
            disp_frame, last_label, last_conf, last_emotion, fps,
            show_debug=show_debug,
        )

        cv2.imshow(WIN_NAME, disp_frame)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[app] Quit requested.")
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"[app] Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"[app] Debug mode: {'ON' if show_debug else 'OFF'}")
        elif key == ord('r'):
            reset_window()
            last_label = "Ready"
            print("[app] Prediction window reset.")

    # ── Clean shutdown ──────────────────────────────────────────────────────
    stop_event.set()
    t1.join(timeout=2); t2.join(timeout=2); t3.join(timeout=2)
    cv2.destroyAllWindows()
    print("[app] Shutdown complete.")

# ═══════════════════════════════════════════════════════════════════════════════
#  Sprint mode — single loop, neutral emotion, smoothing in app layer
# ═══════════════════════════════════════════════════════════════════════════════

SPRINT_WINDOW_SIZE    = 5
SPRINT_MIN_VOTES      = 3
SPRINT_MIN_CONFIDENCE = 0.65
SPRINT_EMOTION        = "neutral"


def _sprint_smooth(window: deque, lbl: str, conf: float) -> tuple[str | None, float]:
    window.append((lbl, conf))
    if len(window) < SPRINT_WINDOW_SIZE:
        return None, 0.0
    labels = [p[0] for p in window]
    confs = [p[1] for p in window]
    top = max(set(labels), key=labels.count)
    if labels.count(top) >= SPRINT_MIN_VOTES and np.mean(confs) >= SPRINT_MIN_CONFIDENCE:
        return top, float(np.mean(confs))
    return None, 0.0


def run_sprint(source):
    """Single-loop desktop app (.cursor/instruction.md sprint architecture)."""
    from src.landmark_gate import gate_from_frame, reset_gate

    print("[sprint] Loading model…")
    load_model()
    print("[sprint] Model ready. Press Q to quit, L for landmarks, D for debug.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[sprint] ERROR: Cannot open source '{source}'")
        return

    pred_window = deque(maxlen=SPRINT_WINDOW_SIZE)
    show_landmarks = True
    show_debug = False
    fps_deque = deque(maxlen=FPS_WINDOW)
    t_last = time.perf_counter()

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 800, 520)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        active, raw = gate_from_frame(frame)

        if raw is None or raw[:126].sum() == 0:
            lbl, conf = "__no_hands__", 0.0
        elif not active:
            lbl, conf = "__still__", 0.0
        else:
            lbl, conf, _ = predict_frame(frame, SPRINT_EMOTION, raw=True)

        if lbl == "__no_hands__":
            status = "No hand detected — move closer"
            colour = C_GREY
            pred_window.clear()
        elif lbl == "__still__" or not active:
            status = "Ready — show a sign"
            colour = C_WHITE
            pred_window.clear()
        else:
            committed, avg_conf = _sprint_smooth(pred_window, lbl, conf)
            if committed:
                status = f"Sign: {committed}  ({avg_conf:.0%})"
                colour = C_GREEN
            elif conf < SPRINT_MIN_CONFIDENCE:
                status = "Detecting..."
                colour = C_YELLOW
            else:
                status = "Detecting..."
                colour = C_YELLOW

        now = time.perf_counter()
        fps_deque.append(now - t_last)
        t_last = now
        fps = 1.0 / (sum(fps_deque) / len(fps_deque)) if fps_deque else 0.0

        disp = frame.copy()
        if show_landmarks:
            mp_results = get_last_mp_results()
            if mp_results is not None:
                draw_landmarks(disp, mp_results)

        h, w = disp.shape[:2]
        draw_rounded_rect(disp, 0, h - 70, w, h, C_OVERLAY, alpha=0.7)
        cv2.putText(disp, status, (20, h - 40), FONT, 0.75, colour, 2, cv2.LINE_AA)
        cv2.putText(
            disp, f"Emotion: {SPRINT_EMOTION}", (20, h - 12),
            FONT, 0.55, C_GREY, 1, cv2.LINE_AA,
        )
        fps_text = f"{fps:.1f} FPS"
        (tw, _), _ = cv2.getTextSize(fps_text, FONT, 0.5, 1)
        cv2.putText(disp, fps_text, (w - tw - 10, 22), FONT, 0.5, C_WHITE, 1, cv2.LINE_AA)

        if show_debug:
            dbg = f"raw: {lbl} {conf:.2f}"
            cv2.putText(disp, dbg, (20, 55), FONT, 0.5, C_YELLOW, 1, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l"):
            show_landmarks = not show_landmarks
        if key == ord("d"):
            show_debug = not show_debug
        if key == ord("r"):
            pred_window.clear()
            reset_gate()
            reset_window()

    cap.release()
    cv2.destroyAllWindows()
    print("[sprint] Shutdown complete.")

# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="ESL Real-Time Sign Language Recognition")
    p.add_argument(
        "--source", default="0",
        help="Camera index (0, 1, …) or path to video file. Default: 0",
    )
    p.add_argument(
        "--sprint",
        action="store_true",
        help="Sprint mode: single loop, neutral emotion, no DeepFace thread",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    if args.sprint:
        run_sprint(source)
    else:
        run(source)
