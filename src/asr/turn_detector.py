"""Smart Turn v3 — audio-based end-of-turn detection via ONNX.

Takes the last 8 seconds of audio and predicts whether the speaker
has finished their turn. 8MB model, ~12ms on CPU.
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

from .config import settings

log = logging.getLogger(__name__)

cfg = settings.turn
SAMPLE_RATE = settings.asr.sample_rate
WINDOW_SAMPLES = int(cfg.audio_window_sec * SAMPLE_RATE)


class SmartTurnDetector:
    """Detects end-of-turn from raw audio using Smart Turn v3 ONNX model."""

    def __init__(self):
        log.info("Loading Smart Turn v3 from %s/%s ...", cfg.model_repo, cfg.model_file)

        model_path = hf_hub_download(
            repo_id=cfg.model_repo,
            filename=cfg.model_file,
        )
        log.info("Model cached at %s", model_path)

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = cfg.cpu_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(model_path, sess_options=so)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)

        log.info("Smart Turn v3 ready (threshold=%.2f, window=%.1fs)",
                 cfg.threshold, cfg.audio_window_sec)

    def predict(self, audio_f32: np.ndarray) -> tuple[bool, float]:
        """Predict whether the speaker's turn is complete.

        Args:
            audio_f32: float32 numpy array of recent audio at 16kHz,
                       ideally the last ~8 seconds of the utterance
                       including trailing silence.

        Returns:
            (is_complete, probability) — bool and raw sigmoid score.
        """
        # Truncate/pad to exactly 8 seconds
        if len(audio_f32) > WINDOW_SAMPLES:
            audio_f32 = audio_f32[-WINDOW_SAMPLES:]
        elif len(audio_f32) < WINDOW_SAMPLES:
            pad = WINDOW_SAMPLES - len(audio_f32)
            audio_f32 = np.pad(audio_f32, (pad, 0), mode="constant", constant_values=0)

        # Whisper feature extraction → 80-channel mel spectrogram
        inputs = self._feature_extractor(
            audio_f32,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )

        features = inputs.input_features.squeeze(0).astype(np.float32)
        features = np.expand_dims(features, axis=0)

        outputs = self._session.run(None, {"input_features": features})
        prob = float(outputs[0][0].item())

        return prob >= cfg.threshold, prob
