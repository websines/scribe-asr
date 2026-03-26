"""Nemotron cache-aware streaming ASR engine.

Wraps NeMo's conformer_stream_step into a per-session object
that accepts int16 PCM chunks and returns transcript text.
"""

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from omegaconf import open_dict

from .config import settings

log = logging.getLogger(__name__)

gpu = settings.gpu
cfg = settings.asr


def _extract_text(hyps) -> list[str]:
    if isinstance(hyps[0], Hypothesis):
        return [h.text for h in hyps]
    return list(hyps)


class ASREngine:
    """Loads Nemotron once. Creates lightweight sessions per connection."""

    def __init__(self):
        # Cap VRAM
        torch.cuda.set_per_process_memory_fraction(
            gpu.vram_limit_gb / gpu.vram_total_gb, device=gpu.device
        )
        self.device = torch.device(f"cuda:{gpu.device}")
        log.info("VRAM capped at %.1f / %.1f GB on %s",
                 gpu.vram_limit_gb, gpu.vram_total_gb, self.device)

        # Load model
        log.info("Loading %s ...", cfg.model_name)
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=cfg.model_name, map_location=self.device
        )
        self.model.to(self.device).eval()
        self.model.freeze()

        # Streaming context
        self.model.encoder.set_default_att_context_size(
            [cfg.left_context, cfg.lookahead_frames]
        )

        # Greedy RNNT decoding (fastest)
        decoding_cfg = self.model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(self.model, "joint"):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
        self.model.change_decoding_strategy(decoding_cfg)

        # Preprocessor (no dither/padding for streaming)
        cfg_copy = copy.deepcopy(self.model.cfg)
        with open_dict(cfg_copy):
            cfg_copy.preprocessor.dither = 0.0
            cfg_copy.preprocessor.pad_to = 0
            cfg_copy.preprocessor.normalize = "None"
        self.preprocessor = EncDecCTCModelBPE.from_config_dict(
            cfg_copy.preprocessor
        ).to(self.device)

        self.pre_encode_cache_size = (
            self.model.encoder.streaming_cfg.pre_encode_cache_size[1]
        )
        self.num_channels = self.model.cfg.preprocessor.features

        log.info("ASR engine ready — chunk=%dms, lookahead=%dms",
                 cfg.chunk_ms, cfg.lookahead_frames * cfg.encoder_step_ms)

    def new_session(self) -> "StreamingSession":
        return StreamingSession(self)


@dataclass
class StreamingSession:
    """Per-connection state. Shares model weights with the engine."""

    engine: ASREngine
    _step: int = 0
    _previous_hypotheses: object = None
    _pred_out_stream: object = None
    _cache_last_channel: torch.Tensor = field(default=None, repr=False)
    _cache_last_time: torch.Tensor = field(default=None, repr=False)
    _cache_last_channel_len: torch.Tensor = field(default=None, repr=False)
    _cache_pre_encode: torch.Tensor = field(default=None, repr=False)
    transcript: str = ""

    def __post_init__(self):
        self.reset()

    def reset(self):
        model = self.engine.model
        device = self.engine.device

        c, t, cl = model.encoder.get_initial_cache_state(batch_size=1)
        self._cache_last_channel = c.to(device) if c is not None else c
        self._cache_last_time = t.to(device) if t is not None else t
        self._cache_last_channel_len = cl.to(device) if cl is not None else cl

        self._cache_pre_encode = torch.zeros(
            (1, self.engine.num_channels, self.engine.pre_encode_cache_size),
            device=device,
        )
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step = 0
        self.transcript = ""

    def process_chunk(self, pcm_int16: np.ndarray) -> str:
        """Feed int16 PCM chunk at 16kHz, return current transcript."""
        engine = self.engine
        device = engine.device

        audio = pcm_int16.astype(np.float32) / 32768.0
        signal = torch.from_numpy(audio).unsqueeze(0).to(device)
        signal_len = torch.tensor([audio.shape[0]], device=device)

        processed, proc_len = engine.preprocessor(
            input_signal=signal, length=signal_len
        )

        processed = torch.cat([self._cache_pre_encode, processed], dim=-1)
        proc_len = proc_len + self._cache_pre_encode.shape[-1]
        self._cache_pre_encode = processed[:, :, -engine.pre_encode_cache_size:]

        with torch.no_grad():
            (
                self._pred_out_stream,
                transcribed_texts,
                self._cache_last_channel,
                self._cache_last_time,
                self._cache_last_channel_len,
                self._previous_hypotheses,
            ) = engine.model.conformer_stream_step(
                processed_signal=processed,
                processed_signal_length=proc_len,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self._previous_hypotheses,
                previous_pred_out=self._pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )

        self._step += 1
        texts = _extract_text(transcribed_texts)
        self.transcript = texts[0] if texts else ""
        return self.transcript
