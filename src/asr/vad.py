"""Silero VAD wrapper — detects speech start/end on 32ms audio chunks."""

import logging

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

from .config import settings

log = logging.getLogger(__name__)

cfg = settings.vad


class SileroVAD:
    """Thin wrapper around Silero VAD for streaming speech detection.

    Feed 512-sample (32ms) chunks of int16 PCM at 16kHz.
    Returns speech events: {"start": ...} or {"end": ...} or None.
    """

    def __init__(self):
        log.info("Loading Silero VAD...")
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=cfg.threshold,
            sampling_rate=settings.asr.sample_rate,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )
        log.info("Silero VAD ready (threshold=%.2f, silence=%dms)",
                 cfg.threshold, cfg.min_silence_ms)

    def process_chunk(self, pcm_int16: np.ndarray) -> dict | None:
        """Feed a 512-sample int16 chunk, returns speech event dict or None.

        Returns:
            {"start": sample_offset} when speech begins
            {"end": sample_offset}   when speech ends
            None                     otherwise
        """
        audio_f32 = torch.from_numpy(
            pcm_int16.astype(np.float32) / 32768.0
        )
        return self._iterator(audio_f32, return_seconds=False)

    def reset(self):
        self._iterator.reset_states()
