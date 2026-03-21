# src/diarizer/embeddings.py
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import normalize

from .utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingExtractor:
    def __init__(self, cfg_emb: DictConfig, cfg_seg: DictConfig, hf_token: str):
        self.cfg_emb    = cfg_emb
        self.cfg_seg    = cfg_seg
        self.hf_token   = hf_token
        self._inference = None
        logger.debug("EmbeddingExtractor initialised — model: %s", cfg_emb.model)

    def _load(self) -> None:
        if self._inference is not None:
            return
        from pyannote.audio import Inference, Model

        logger.info("Loading embedding model: %s", self.cfg_emb.model)
        t0    = time.perf_counter()
        model = Model.from_pretrained(
            self.cfg_emb.model, use_auth_token=self.hf_token
        )
        self._inference = Inference(model, window="whole")
        logger.info("Embedding model loaded in %.2fs", time.perf_counter() - t0)

    def extract(
        self,
        wav_path: str,
        segments: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        self._load()
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

        if not embeddings:
            raise RuntimeError(
                "No valid embeddings extracted. "
                "Check audio quality and min_segment_s."
            )

        emb_array = np.array(embeddings)
        if self.cfg_emb.normalize:
            emb_array = normalize(emb_array, norm="l2")

        logger.info("Embeddings: %d extracted | %d skipped | shape %s | %.2fs",
                    len(embeddings), skipped, emb_array.shape,
                    time.perf_counter() - t0)
        return emb_array, valid_segments

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def pairwise_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        normed = normalize(embeddings, norm="l2")
        return normed @ normed.T