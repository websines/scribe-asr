"""ASR server — FastAPI with HTTP + WebSocket + WebRTC.

Transports:
    WS      /v1/ws/transcribe          — streaming audio via WebSocket
    WebRTC  /v1/webrtc/offer           — SDP offer/answer exchange
            /v1/webrtc/close           — tear down session
            /v1/webrtc/sessions        — list active WebRTC sessions

REST:
    GET  /healthz                      — liveness probe
    GET  /readyz                       — readiness (models loaded)
    GET  /v1/info                      — server config and capabilities
    GET  /v1/sessions                  — active session counts (WS + WebRTC)
    POST /v1/transcribe                — one-shot file transcription

WebSocket protocol (binary + JSON):
    Client → Server:
        binary:  raw int16 PCM at 16kHz mono
        json:    {"type": "reset"}     — new utterance

    Server → Client:
        json:    {"type": "ready"}
                 {"type": "partial",     "text": "...", "step": N}
                 {"type": "final",       "text": "...", "step": N, "turn_probability": 0.95}
                 {"type": "vad_start"}
                 {"type": "vad_end"}
                 {"type": "turn_result", "turn_probability": 0.85}
                 {"type": "error",       "message": "..."}

WebRTC protocol:
    Client sends audio track (any codec — Opus, PCM, etc.)
    Server sends JSON messages via RTCDataChannel "transcription"
    Same event format as WebSocket.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .pipeline import Pipeline, PipelineSession
from .text_fixer import fix_code_terms

log = logging.getLogger(__name__)

# --- Global state ---
_pipeline: Optional[Pipeline] = None
_ready = False
_active_sessions: dict[str, float] = {}   # session_id → connect_time
_boot_time: float = 0


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _ready, _boot_time
    _boot_time = time.time()
    log.info("Loading pipeline models...")
    _pipeline = Pipeline()
    _ready = True
    log.info("Server ready.")
    yield
    _ready = False
    log.info("Shutting down.")


app = FastAPI(
    title="ASR Streaming Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount WebRTC router
from .webrtc import router as webrtc_router, _peers as _webrtc_peers
app.include_router(webrtc_router)


# ──────────────────────────── REST Endpoints ────────────────────────────


class HealthResponse(BaseModel):
    status: str
    uptime_s: float = 0


@app.get("/healthz")
async def healthz() -> HealthResponse:
    """Liveness — always returns 200 if process is running."""
    return HealthResponse(status="alive", uptime_s=time.time() - _boot_time)


@app.get("/readyz")
async def readyz():
    """Readiness — 200 only after models are loaded."""
    if not _ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading"},
        )
    return HealthResponse(status="ready", uptime_s=time.time() - _boot_time)


class InfoResponse(BaseModel):
    model: str
    sample_rate: int
    chunk_ms: int
    lookahead_ms: int
    vram_limit_gb: float
    max_connections: int
    active_connections: int


@app.get("/v1/info")
async def info() -> InfoResponse:
    return InfoResponse(
        model=settings.asr.model_name,
        sample_rate=settings.asr.sample_rate,
        chunk_ms=settings.asr.chunk_ms,
        lookahead_ms=settings.asr.lookahead_frames * settings.asr.encoder_step_ms,
        vram_limit_gb=settings.gpu.vram_limit_gb,
        max_connections=settings.server.max_connections,
        active_connections=len(_active_sessions) + len(_webrtc_peers),
    )


class SessionsResponse(BaseModel):
    active: int
    max: int
    sessions: list[dict]


@app.get("/v1/sessions")
async def sessions() -> SessionsResponse:
    now = time.time()
    ws_sessions = [
        {"id": sid, "transport": "ws", "duration_s": round(now - t, 1)}
        for sid, t in _active_sessions.items()
    ]
    rtc_sessions = [
        {"id": sid, "transport": "webrtc", "state": p["pc"].connectionState,
         "duration_s": round(now - p["created"], 1)}
        for sid, p in _webrtc_peers.items()
    ]
    total = len(_active_sessions) + len(_webrtc_peers)
    return SessionsResponse(
        active=total,
        max=settings.server.max_connections,
        sessions=ws_sessions + rtc_sessions,
    )


class TranscribeResponse(BaseModel):
    text: str
    duration_s: float
    processing_s: float


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
):
    """One-shot file transcription (non-streaming). For testing and batch use."""
    if not _ready:
        return JSONResponse(status_code=503, content={"error": "models loading"})

    import soundfile as sf
    import io

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # Convert to mono 16kHz int16
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != settings.asr.sample_rate:
        duration = len(audio) / sr
        n = int(duration * settings.asr.sample_rate)
        x_old = np.linspace(0, duration, len(audio), endpoint=False)
        x_new = np.linspace(0, duration, n, endpoint=False)
        audio = np.interp(x_new, x_old, audio)

    pcm = (audio * 32768).clip(-32768, 32767).astype(np.int16)

    t0 = time.time()
    session = _pipeline.new_session()
    chunk_size = settings.asr.chunk_samples

    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        if len(chunk) < chunk_size:
            padded = np.zeros(chunk_size, dtype=np.int16)
            padded[:len(chunk)] = chunk
            chunk = padded
        session._asr_session.process_chunk(chunk)

    text = fix_code_terms(session._asr_session.transcript)
    proc_time = time.time() - t0

    return TranscribeResponse(
        text=text,
        duration_s=round(len(pcm) / settings.asr.sample_rate, 2),
        processing_s=round(proc_time, 3),
    )


# ──────────────────────────── WebSocket Endpoint ────────────────────────────


@app.websocket("/v1/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    if not _ready:
        await ws.close(code=1013, reason="models loading")
        return

    total = len(_active_sessions) + len(_webrtc_peers)
    if total >= settings.server.max_connections:
        await ws.accept()
        await ws.send_json({"type": "error", "message": "server full"})
        await ws.close()
        return

    await ws.accept()

    session_id = f"{ws.client.host}:{ws.client.port}-{int(time.time()*1000)}"
    _active_sessions[session_id] = time.time()
    log.info("WS connected: %s (active=%d)", session_id, len(_active_sessions))

    pipeline_session = _pipeline.new_session()
    await ws.send_json({"type": "ready"})

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                # Binary frame — raw int16 PCM audio
                pcm = np.frombuffer(message["bytes"], dtype=np.int16)

                events = await asyncio.to_thread(
                    pipeline_session.feed_audio, pcm
                )

                for event in events:
                    await ws.send_json({
                        "type": event.type,
                        "text": event.text,
                        "is_final": event.is_final,
                        "step": event.step,
                        "turn_probability": round(event.turn_probability, 4),
                        "timestamp": event.timestamp,
                    })

            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "reset":
                    final = pipeline_session.force_finalize()
                    if final:
                        await ws.send_json({
                            "type": final.type,
                            "text": final.text,
                            "is_final": True,
                            "step": final.step,
                            "turn_probability": 0,
                            "timestamp": final.timestamp,
                        })
                    pipeline_session.reset()

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Error in session %s", session_id)
    finally:
        # Cleanup
        final = pipeline_session.force_finalize()
        _active_sessions.pop(session_id, None)
        log.info("WS disconnected: %s (active=%d)", session_id, len(_active_sessions))


# ──────────────────────────── Entry Point ────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log.info("Starting ASR server on %s:%d", settings.server.host, settings.server.port)
    uvicorn.run(
        "asr.server:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
        workers=1,           # single worker — GPU model is shared in-process
    )
