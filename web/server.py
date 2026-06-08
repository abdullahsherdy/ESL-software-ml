"""
web/server.py — FastAPI WebSocket server (Phase 4, optional).
Decode browser JPEG → BGR → predict_frame(frame, "neutral") for CPU-only demo.

Run: uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from src.inference import load_model, predict_frame

WEB_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_ready = False


@app.on_event("startup")
def _startup_load_model() -> None:
    global _model_ready
    try:
        load_model()
        _model_ready = True
    except Exception as e:
        print(f"[web] Model not loaded: {e}")
        _model_ready = False


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_ready}


@app.get("/", response_class=HTMLResponse)
def index():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/manifest.json")
def manifest():
    return FileResponse(WEB_DIR / "manifest.json", media_type="application/json")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            t0 = time.perf_counter()

            try:
                raw = base64.b64decode(data)
            except Exception:
                raw = base64.b64decode(data.split(",")[-1])

            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text(json.dumps({
                    "label": "Error",
                    "confidence": 0.0,
                    "emotion": "neutral",
                    "latency_ms": None,
                    "fps_hint": None,
                    "error": "decode_failed",
                }))
                continue

            if not _model_ready:
                await websocket.send_text(json.dumps({
                    "label": "Error",
                    "confidence": 0.0,
                    "emotion": "neutral",
                    "latency_ms": None,
                    "fps_hint": None,
                    "error": "model_not_loaded",
                }))
                continue

            try:
                lbl, conf, emotion = predict_frame(frame, cached_emotion="neutral")
            except Exception as exc:
                print(f"[web] predict_frame error: {exc}")
                await websocket.send_text(json.dumps({
                    "label": "Error",
                    "confidence": 0.0,
                    "emotion": "neutral",
                    "latency_ms": None,
                    "fps_hint": None,
                    "error": "inference_failed",
                }))
                continue

            latency_ms = (time.perf_counter() - t0) * 1000.0
            await websocket.send_text(json.dumps({
                "label":      lbl if lbl not in ("No hand detected",) else lbl,
                "confidence": round(float(conf), 4),
                "emotion":    emotion,
                "latency_ms": round(latency_ms, 2),
                "fps_hint":   round(1000.0 / latency_ms, 1) if latency_ms > 0 else None,
            }))
    except WebSocketDisconnect:
        pass
