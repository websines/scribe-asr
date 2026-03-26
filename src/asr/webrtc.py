"""WebRTC signaling + audio track handler.

Flow:
    1. Client creates RTCPeerConnection in browser
    2. Client POSTs SDP offer to /v1/webrtc/offer
    3. Server creates answer, returns SDP
    4. WebRTC connection established (DTLS/SRTP)
    5. Audio track decoded to PCM, fed into pipeline
    6. Transcription results sent back via data channel

Endpoints (mounted on the FastAPI app by server.py):
    POST /v1/webrtc/offer   — SDP offer/answer exchange
    POST /v1/webrtc/close   — tear down a session
"""

import asyncio
import json
import logging
import time
import uuid

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
from fastapi import APIRouter

from .pipeline import Pipeline, PipelineSession

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/webrtc", tags=["webrtc"])

# Track active peer connections
_peers: dict[str, dict] = {}
_relay = MediaRelay()


def _get_pipeline() -> Pipeline:
    """Deferred import to avoid circular deps — pipeline set by server.py."""
    from .server import _pipeline
    return _pipeline


class AudioTrackProcessor:
    """Receives decoded audio frames from a WebRTC track,
    resamples to 16kHz mono int16, feeds into the ASR pipeline,
    and sends results back via the data channel."""

    def __init__(self, session: PipelineSession, data_channel):
        self._session = session
        self._dc = data_channel
        self._running = True

    async def process_track(self, track):
        """Consume audio frames from the WebRTC track."""
        import av.audio.resampler as avr

        resampler = None

        while self._running:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            # Decode to numpy — frame is an av.AudioFrame
            arr = frame.to_ndarray()  # shape: (channels, samples), dtype float/int

            # Init resampler on first frame (match source format → 16kHz mono s16)
            if resampler is None:
                resampler = av.AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=16000,
                )

            # Resample
            resampled_frames = resampler.resample(frame)
            for rf in resampled_frames:
                pcm = rf.to_ndarray().flatten().astype(np.int16)

                if len(pcm) == 0:
                    continue

                events = await asyncio.to_thread(
                    self._session.feed_audio, pcm
                )

                for event in events:
                    if self._dc and self._dc.readyState == "open":
                        self._dc.send(json.dumps({
                            "type": event.type,
                            "text": event.text,
                            "is_final": event.is_final,
                            "step": event.step,
                            "turn_probability": round(event.turn_probability, 4),
                            "timestamp": event.timestamp,
                        }))

    def stop(self):
        self._running = False


@router.post("/offer")
async def webrtc_offer(request_data: dict):
    """Exchange SDP offer/answer to establish WebRTC connection.

    Request body:
        {"sdp": "...", "type": "offer"}

    Response:
        {"sdp": "...", "type": "answer", "session_id": "..."}
    """
    pipeline = _get_pipeline()
    if pipeline is None:
        return {"error": "models loading"}, 503

    offer = RTCSessionDescription(
        sdp=request_data["sdp"],
        type=request_data["type"],
    )

    pc = RTCPeerConnection(
        configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
        )
    )
    session_id = uuid.uuid4().hex[:12]
    pipeline_session = pipeline.new_session()
    processor = None

    # Create data channel for sending results back to client
    dc = pc.createDataChannel("transcription", ordered=True)

    @dc.on("open")
    def on_dc_open():
        log.info("Data channel open for %s", session_id)
        dc.send(json.dumps({"type": "ready"}))

    @pc.on("track")
    async def on_track(track):
        nonlocal processor
        if track.kind != "audio":
            return

        log.info("Audio track received for %s", session_id)
        processor = AudioTrackProcessor(pipeline_session, dc)
        asyncio.ensure_future(processor.process_track(track))

        @track.on("ended")
        async def on_track_ended():
            log.info("Audio track ended for %s", session_id)
            final = pipeline_session.force_finalize()
            if final and dc.readyState == "open":
                dc.send(json.dumps({
                    "type": final.type,
                    "text": final.text,
                    "is_final": True,
                    "step": final.step,
                    "turn_probability": 0,
                    "timestamp": final.timestamp,
                }))

    @pc.on("connectionstatechange")
    async def on_connection_state():
        log.info("WebRTC %s state: %s", session_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            if processor:
                processor.stop()
            pipeline_session.force_finalize()
            _peers.pop(session_id, None)
            log.info("WebRTC %s cleaned up (active=%d)", session_id, len(_peers))

    # Set remote offer, create answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    _peers[session_id] = {
        "pc": pc,
        "session": pipeline_session,
        "processor": processor,
        "created": time.time(),
    }
    log.info("WebRTC %s established (active=%d)", session_id, len(_peers))

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session_id,
    }


@router.post("/close")
async def webrtc_close(request_data: dict):
    """Tear down a WebRTC session.

    Request body:
        {"session_id": "..."}
    """
    session_id = request_data.get("session_id", "")
    peer = _peers.pop(session_id, None)
    if peer:
        if peer.get("processor"):
            peer["processor"].stop()
        peer["session"].force_finalize()
        await peer["pc"].close()
        return {"status": "closed"}
    return {"status": "not_found"}


@router.get("/sessions")
async def webrtc_sessions():
    """List active WebRTC sessions."""
    now = time.time()
    return {
        "active": len(_peers),
        "sessions": [
            {"id": sid, "state": p["pc"].connectionState, "duration_s": round(now - p["created"], 1)}
            for sid, p in _peers.items()
        ],
    }
