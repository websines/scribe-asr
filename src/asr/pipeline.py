"""Full ASR pipeline: Silero VAD → Nemotron ASR → Smart Turn v3.

State machine per session:

    IDLE ──(VAD: speech start)──→ LISTENING
    LISTENING ──(each ASR chunk)──→ emit partial
    LISTENING ──(VAD: speech end)──→ TURN_CHECK
    TURN_CHECK ──(Smart Turn: done)──→ emit final → IDLE
    TURN_CHECK ──(Smart Turn: not done)──→ LISTENING
    TURN_CHECK ──(VAD: speech start)──→ LISTENING
"""

import enum
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .engine import ASREngine, StreamingSession
from .text_fixer import fix_code_terms
from .turn_detector import SmartTurnDetector
from .vad import SileroVAD

log = logging.getLogger(__name__)

SAMPLE_RATE = settings.asr.sample_rate
VAD_WINDOW = settings.vad.window_samples             # 512 samples = 32ms
ASR_CHUNK = settings.asr.chunk_samples                # e.g. 8960 samples = 560ms
TURN_WINDOW = int(settings.turn.audio_window_sec * SAMPLE_RATE)  # 128000 = 8s


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TURN_CHECK = "turn_check"


@dataclass
class PipelineEvent:
    """Events emitted by the pipeline."""
    type: str                      # "partial" | "final" | "vad_start" | "vad_end" | "turn_result"
    text: str = ""
    is_final: bool = False
    turn_probability: float = 0.0
    step: int = 0
    timestamp: float = field(default_factory=time.time)


class Pipeline:
    """Shared models — loaded once at server start."""

    def __init__(self):
        log.info("Initializing pipeline models...")
        self.vad = SileroVAD()
        self.asr_engine = ASREngine()
        self.turn_detector = SmartTurnDetector()
        log.info("Pipeline ready.")

    def new_session(self) -> "PipelineSession":
        return PipelineSession(self)


class PipelineSession:
    """Per-connection pipeline state. Feed small audio frames, get events back."""

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self._asr_session: StreamingSession = pipeline.asr_engine.new_session()
        self._state = State.IDLE

        # Audio buffers
        self._vad_buf = np.array([], dtype=np.int16)    # accumulate to VAD_WINDOW
        self._asr_buf = np.array([], dtype=np.int16)    # accumulate to ASR_CHUNK
        self._turn_buf = np.array([], dtype=np.float32)  # rolling window for Smart Turn

    def reset(self):
        """Reset for a new utterance."""
        self._asr_session.reset()
        self._pipeline.vad.reset()
        self._state = State.IDLE
        self._vad_buf = np.array([], dtype=np.int16)
        self._asr_buf = np.array([], dtype=np.int16)
        self._turn_buf = np.array([], dtype=np.float32)

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def transcript(self) -> str:
        return self._asr_session.transcript

    def feed_audio(self, pcm_int16: np.ndarray) -> list[PipelineEvent]:
        """Feed raw int16 PCM at 16kHz. Returns list of events (may be empty).

        Audio can be any size — internally chunked to VAD/ASR windows.
        """
        events: list[PipelineEvent] = []

        # Append to VAD buffer
        self._vad_buf = np.concatenate([self._vad_buf, pcm_int16])

        # Process in VAD-sized windows (512 samples = 32ms)
        while len(self._vad_buf) >= VAD_WINDOW:
            vad_chunk = self._vad_buf[:VAD_WINDOW]
            self._vad_buf = self._vad_buf[VAD_WINDOW:]

            vad_event = self._pipeline.vad.process_chunk(vad_chunk)

            # Maintain rolling float32 buffer for Smart Turn
            chunk_f32 = vad_chunk.astype(np.float32) / 32768.0
            self._turn_buf = np.concatenate([self._turn_buf, chunk_f32])
            if len(self._turn_buf) > TURN_WINDOW:
                self._turn_buf = self._turn_buf[-TURN_WINDOW:]

            # --- State transitions ---
            if self._state == State.IDLE:
                if vad_event and "start" in vad_event:
                    self._state = State.LISTENING
                    self._asr_session.reset()
                    events.append(PipelineEvent(type="vad_start"))

            elif self._state == State.LISTENING:
                # Accumulate audio for ASR
                self._asr_buf = np.concatenate([self._asr_buf, vad_chunk])

                # Process complete ASR chunks
                while len(self._asr_buf) >= ASR_CHUNK:
                    asr_chunk = self._asr_buf[:ASR_CHUNK]
                    self._asr_buf = self._asr_buf[ASR_CHUNK:]

                    text = self._asr_session.process_chunk(asr_chunk)
                    text = fix_code_terms(text)

                    events.append(PipelineEvent(
                        type="partial",
                        text=text,
                        step=self._asr_session._step,
                    ))

                # Check for speech end
                if vad_event and "end" in vad_event:
                    self._state = State.TURN_CHECK
                    events.append(PipelineEvent(type="vad_end"))

                    # Flush remaining ASR buffer (pad with silence)
                    if len(self._asr_buf) > 0:
                        padded = np.zeros(ASR_CHUNK, dtype=np.int16)
                        padded[:len(self._asr_buf)] = self._asr_buf
                        self._asr_buf = np.array([], dtype=np.int16)

                        text = self._asr_session.process_chunk(padded)
                        text = fix_code_terms(text)

                        events.append(PipelineEvent(
                            type="partial",
                            text=text,
                            step=self._asr_session._step,
                        ))

                    # Evaluate turn completion
                    is_done, prob = self._pipeline.turn_detector.predict(self._turn_buf)
                    events.append(PipelineEvent(
                        type="turn_result",
                        turn_probability=prob,
                    ))

                    if is_done:
                        events.append(PipelineEvent(
                            type="final",
                            text=fix_code_terms(self._asr_session.transcript),
                            is_final=True,
                            step=self._asr_session._step,
                            turn_probability=prob,
                        ))
                        self._state = State.IDLE
                        self._asr_session.reset()
                        self._pipeline.vad.reset()
                        self._asr_buf = np.array([], dtype=np.int16)
                    else:
                        # Not done — speaker is just pausing, keep listening
                        self._state = State.LISTENING

            elif self._state == State.TURN_CHECK:
                # We might get here if VAD fires start again quickly
                if vad_event and "start" in vad_event:
                    self._state = State.LISTENING
                    events.append(PipelineEvent(type="vad_start"))

        return events

    def force_finalize(self) -> PipelineEvent | None:
        """Force-emit a final transcript (e.g. client disconnecting)."""
        if self._state == State.IDLE and not self._asr_session.transcript:
            return None

        # Flush remaining audio
        if len(self._asr_buf) > 0:
            padded = np.zeros(ASR_CHUNK, dtype=np.int16)
            padded[:len(self._asr_buf)] = self._asr_buf
            self._asr_buf = np.array([], dtype=np.int16)
            self._asr_session.process_chunk(padded)

        text = fix_code_terms(self._asr_session.transcript)
        self.reset()

        if text:
            return PipelineEvent(
                type="final",
                text=text,
                is_final=True,
            )
        return None
