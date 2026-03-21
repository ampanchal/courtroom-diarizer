# src/diarizer/audio.py
import hashlib
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig

from .utils.logging import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger.debug("AudioProcessor initialised — target: %d Hz mono", cfg.sample_rate)

    def process(self, path: str) -> Tuple[torch.Tensor, int, dict]:
        t0   = time.perf_counter()
        path = Path(path)
        logger.info("Processing audio: %s", path.name)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sr = self._load(path)
        meta = {
            "filename":           path.name,
            "file_hash":          self._sha256(path),
            "original_sr":        sr,
            "original_channels":  waveform.shape[0],
            "warnings":           [],
        }

        warnings = self._validate(waveform, sr, path)
        meta["warnings"].extend(warnings)

        if sr != self.cfg.sample_rate:
            waveform = self._resample(waveform, sr, self.cfg.sample_rate)
            logger.debug("  Resampled %d -> %d Hz", sr, self.cfg.sample_rate)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            logger.debug("  %d channels -> mono", meta["original_channels"])

        waveform = self._highpass(waveform, self.cfg.sample_rate, self.cfg.highpass_hz)
        waveform = self._normalize_lufs(waveform, self.cfg.sample_rate)

        clip_pct = self._clipping_pct(waveform)
        meta["clip_pct"] = round(float(clip_pct), 4)
        if clip_pct > self.cfg.max_clip_pct:
            msg = f"High clipping detected: {clip_pct:.2f}%"
            meta["warnings"].append(msg)
            logger.warning("  %s in %s", msg, path.name)

        meta["duration_s"] = round(waveform.shape[-1] / self.cfg.sample_rate, 3)
        meta["elapsed_s"]  = round(time.perf_counter() - t0, 3)

        logger.info("  OK %s | %.1fs | %s",
                    path.name, meta["duration_s"],
                    "clean" if not meta["warnings"] else f"{len(meta['warnings'])} warnings")
        return waveform, self.cfg.sample_rate, meta

    def _load(self, path: Path) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sr = torchaudio.load(str(path))
            return waveform, sr
        except Exception:
            data, sr = sf.read(str(path), always_2d=True)
            return torch.from_numpy(data.T).float(), sr

    def _validate(self, waveform: torch.Tensor, sr: int, path: Path) -> list:
        warnings = []
        duration = waveform.shape[-1] / sr
        if duration < self.cfg.min_duration_s:
            warnings.append(f"Very short audio: {duration:.2f}s")
        rms = waveform.pow(2).mean().sqrt().item()
        if rms < 1e-4:
            warnings.append("Audio appears near-silent")
        return warnings

    def _resample(self, waveform, orig_sr, target_sr):
        return T.Resample(orig_freq=orig_sr, new_freq=target_sr)(waveform)

    def _highpass(self, waveform, sr, cutoff_hz):
        return torchaudio.functional.highpass_biquad(
            waveform, sample_rate=sr, cutoff_freq=float(cutoff_hz)
        )

    def _normalize_lufs(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        target_db   = self.cfg.target_lufs
        rms         = waveform.pow(2).mean().sqrt()
        if rms < 1e-8:
            return waveform
        current_db  = 20 * torch.log10(rms)
        gain_linear = 10 ** ((target_db - current_db.item()) / 20)
        return torch.clamp(waveform * gain_linear, -1.0, 1.0)

    @staticmethod
    def _clipping_pct(waveform: torch.Tensor) -> float:
        return float((waveform.abs() >= 0.999).float().mean() * 100)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]