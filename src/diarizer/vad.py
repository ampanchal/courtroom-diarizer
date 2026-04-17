# src/diarizer/vad.py
import time
from pathlib import Path
from typing import List, Tuple

import torch
from omegaconf import DictConfig

from .utils.logging import get_logger

logger = get_logger(__name__)


class VADProcessor:
    def __init__(self, cfg: DictConfig, hf_token: str):
        self.cfg      = cfg
        self.hf_token = hf_token
        self._pipeline = None
        logger.debug("VADProcessor initialised — model: %s", cfg.model)

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection

        logger.info("Loading VAD model: %s", self.cfg.model)
        t0    = time.perf_counter()
        model = Model.from_pretrained(self.cfg.model, use_auth_token=self.hf_token)
        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate({
            "min_duration_on":  self.cfg.min_duration_on,
            "min_duration_off": self.cfg.min_duration_off,
        })
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
        self._pipeline = pipeline
        logger.info("VAD model loaded in %.2fs", time.perf_counter() - t0)

    def detect(self, wav_path: str) -> List[Tuple[float, float]]:
        self._load()
        t0     = time.perf_counter()
        result = self._pipeline(wav_path)

        segments = []
        for segment, _, _ in result.itertracks(yield_label=True):
            segments.append((round(segment.start, 3), round(segment.end, 3)))

        total_speech = sum(e - s for s, e in segments)
        logger.info("VAD: %d segments | %.1fs speech | %.2fs elapsed",
                    len(segments), total_speech, time.perf_counter() - t0)

        if not segments:
            logger.warning("No speech detected — check audio quality and VAD thresholds")

        return segments

    def speech_ratio(self, segments: List[Tuple[float, float]], total_duration: float) -> float:
        if total_duration <= 0:
            return 0.0
        return round(sum(e - s for s, e in segments) / total_duration, 4)