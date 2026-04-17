# src/diarizer/embeddings.py
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import normalize

from .utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingExtractor:
    """
    Embedding backends:
        pyannote — ECAPA-TDNN via pyannote/embedding. DEFAULT.
                   5.29% mean DER on VoxConverse (216 files).
        wavlm    — WavLM-base-plus-sv via SpeechBrain. EXPERIMENTAL.
                   22.37% mean DER on VoxConverse (216 files).
                   Gap is in clustering, not embedding quality.
                   Revisit after threshold optimisation or fine-tuning.
    """

    def __init__(self, cfg_emb: DictConfig, cfg_seg: DictConfig, hf_token: str):
        self.cfg_emb    = cfg_emb
        self.cfg_seg    = cfg_seg
        self.hf_token   = hf_token
        self._inference = None   # pyannote backend
        self._wavlm     = None   # wavlm backend
        self.backend    = getattr(cfg_emb, "backend", "pyannote")
        logger.info("EmbeddingExtractor — backend=%s", self.backend)

    # ── Lazy loaders ──────────────────────────────────────────

    def _load_pyannote(self) -> None:
        if self._inference is not None:
            return
        from pyannote.audio import Inference, Model
        logger.info("Loading pyannote embedding model: %s", self.cfg_emb.model)
        t0    = time.perf_counter()
        model = Model.from_pretrained(
            self.cfg_emb.model, use_auth_token=self.hf_token
        )
        self._inference = Inference(model, window="whole")
        logger.info("Pyannote embedding model loaded in %.2fs",
                    time.perf_counter() - t0)

    def _load_wavlm(self) -> None:
        if self._wavlm is not None:
            return

        # Windows symlink fix — patch pathlib.Path.symlink_to to copy instead
        import pathlib
        import shutil
        original_symlink = pathlib.Path.symlink_to

        def copy_instead_of_symlink(self_path, target, target_is_directory=False):
            try:
                original_symlink(self_path, target, target_is_directory)
            except OSError:
                target_path = pathlib.Path(target)
                if target_path.is_dir():
                    if self_path.exists():
                        shutil.rmtree(self_path)
                    shutil.copytree(target_path, self_path)
                else:
                    shutil.copy2(target_path, self_path)

        pathlib.Path.symlink_to = copy_instead_of_symlink

        from speechbrain.inference.speaker import SpeakerRecognition
        source = getattr(self.cfg_emb, "wavlm_source",
                         "speechbrain/spkrec-ecapa-voxceleb")
        logger.info("Loading WavLM+ECAPA model: %s", source)
        t0 = time.perf_counter()
        self._wavlm = SpeakerRecognition.from_hparams(
            source   = source,
            savedir  = ".cache/wavlm",
            run_opts = {"device": "cpu"},
        )

        # Restore original symlink behaviour
        pathlib.Path.symlink_to = original_symlink
        logger.info("WavLM+ECAPA model loaded in %.2fs",
                    time.perf_counter() - t0)

    # ── Public API ────────────────────────────────────────────

    def extract(
        self,
        wav_path: str,
        segments: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Extract one embedding per speech segment.

        Automatically routes to the configured backend.
        Both backends return L2-normalised 192-d vectors
        so the clustering stage is backend-agnostic.
        """
        if self.backend == "wavlm":
            return self._extract_wavlm(wav_path, segments)
        return self._extract_pyannote(wav_path, segments)

    # ── Pyannote backend ──────────────────────────────────────

    def _extract_pyannote(
        self,
        wav_path: str,
        segments: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        self._load_pyannote()
        from pyannote.core import Segment

        t0             = time.perf_counter()
        embeddings     = []
        valid_segments = []
        skipped        = 0

        for start, end in segments:
            if (end - start) < self.cfg_seg.min_segment_s:
                skipped += 1
                continue
            try:
                emb = self._inference.crop(wav_path, Segment(start, end))
                embeddings.append(emb.squeeze())
                valid_segments.append((start, end))
            except Exception as exc:
                logger.warning("Skipping %.2f-%.2fs: %s", start, end, exc)
                skipped += 1

        return self._finalise(embeddings, valid_segments, skipped,
                              "pyannote", time.perf_counter() - t0)

    # ── WavLM + ECAPA backend ─────────────────────────────────

    def _extract_wavlm(
        self,
        wav_path: str,
        segments: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        WavLM-Base-Plus with ECAPA-TDNN speaker verification head.

        Key advantage over pyannote backend for short utterances:
        WavLM operates on raw waveform samples — it learned its own
        frame-level features from 94k hrs of unlabelled audio via
        masked prediction, so it extracts richer representations
        from fewer frames than hand-crafted mel-filterbanks allow.

        The ECAPA-TDNN head then pools those frame representations
        into a fixed 192-d embedding using attentive statistics
        pooling — which weights frames by their speaker-discriminative
        content rather than averaging blindly.
        """
        self._load_wavlm()
        import torchaudio

        t0             = time.perf_counter()
        embeddings     = []
        valid_segments = []
        skipped        = 0

        # Load full waveform once — crop per segment
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        for start, end in segments:
            duration = end - start

            # WavLM is more robust on short clips but still needs
            # at least 0.3s — pad shorter clips with zeros
            min_dur = max(self.cfg_seg.min_segment_s, 0.3)
            if duration < min_dur:
                skipped += 1
                continue

            try:
                start_sample = int(start * sr)
                end_sample   = int(end   * sr)
                clip = waveform[:, start_sample:end_sample]

                # Pad if clip is shorter than 0.5s (WavLM minimum)
                min_samples = int(0.5 * sr)
                if clip.shape[1] < min_samples:
                    pad = torch.zeros(1, min_samples - clip.shape[1])
                    clip = torch.cat([clip, pad], dim=1)

                # SpeechBrain expects (batch, time)
                emb = self._wavlm.encode_batch(clip)
                embeddings.append(emb.squeeze().detach().numpy())
                valid_segments.append((start, end))

            except Exception as exc:
                logger.warning("WavLM skip %.2f-%.2fs: %s", start, end, exc)
                skipped += 1

        return self._finalise(embeddings, valid_segments, skipped,
                              "wavlm", time.perf_counter() - t0)

    # ── Shared finalisation ───────────────────────────────────

    def _finalise(
        self,
        embeddings:     list,
        valid_segments: list,
        skipped:        int,
        backend:        str,
        elapsed:        float,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        if not embeddings:
            raise RuntimeError(
                "No valid embeddings extracted. "
                "Check audio quality and min_segment_s."
            )
        emb_array = np.array(embeddings)
        if self.cfg_emb.normalize:
            emb_array = normalize(emb_array, norm="l2")
        logger.info(
            "[%s] Embeddings: %d extracted | %d skipped | shape %s | %.2fs",
            backend, len(embeddings), skipped,
            emb_array.shape, elapsed,
        )
        return emb_array, valid_segments

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )

    def pairwise_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        normed = normalize(embeddings, norm="l2")
        return normed @ normed.T