# src/diarizer/clustering.py
import time
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import normalize

from .utils.logging import get_logger

logger = get_logger(__name__)


class SpeakerClusterer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger.debug("SpeakerClusterer initialised — method: %s", cfg.method)

    def cluster(
        self,
        embeddings:   np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> np.ndarray:
        if len(embeddings) == 0:
            raise ValueError("Cannot cluster empty embedding array.")
        if len(embeddings) == 1:
            logger.warning("Only 1 segment — returning single speaker label.")
            return np.array([0])

        t0     = time.perf_counter()
        method = self.cfg.method.lower()

        if method == "ahc":
            labels = self._ahc(embeddings, num_speakers)
        elif method == "spectral":
            labels = self._spectral(embeddings, num_speakers)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'ahc' or 'spectral'.")

        n_speakers = len(set(labels))
        logger.info("Clustering (%s): %d segments -> %d speakers | %.3fs",
                    method, len(embeddings), n_speakers, time.perf_counter() - t0)
        return labels

    def _ahc(self, embeddings: np.ndarray, num_speakers: Optional[int]) -> np.ndarray:
        if num_speakers is not None:
            model = AgglomerativeClustering(
                n_clusters = num_speakers,
                metric     = self.cfg.metric,
                linkage    = self.cfg.linkage,
            )
        else:
            model = AgglomerativeClustering(
                n_clusters         = None,
                distance_threshold = self.cfg.distance_threshold,
                metric             = self.cfg.metric,
                linkage            = self.cfg.linkage,
            )
        return model.fit_predict(embeddings)

    def _spectral(self, embeddings: np.ndarray, num_speakers: Optional[int]) -> np.ndarray:
        sigma    = self.cfg.spectral_sigma
        max_k    = self.cfg.max_speakers
        normed   = normalize(embeddings, norm="l2")
        sim_mat  = normed @ normed.T
        affinity = np.exp(-(1 - sim_mat) / (2 * sigma ** 2))
        np.fill_diagonal(affinity, 1.0)

        if num_speakers is None:
            num_speakers = self._estimate_k(affinity, max_k)
            logger.debug("Spectral: eigenvalue gap -> K=%d", num_speakers)

        model = SpectralClustering(
            n_clusters    = num_speakers,
            affinity      = "precomputed",
            assign_labels = "kmeans",
            random_state  = 42,
            n_init        = 10,
        )
        return model.fit_predict(affinity)

    @staticmethod
    def _estimate_k(affinity: np.ndarray, max_k: int) -> int:
        degree     = affinity.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
        laplacian  = np.eye(len(affinity)) - d_inv_sqrt @ affinity @ d_inv_sqrt
        n          = min(max_k + 1, len(affinity))
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian)[:n])
        gaps        = np.diff(eigenvalues)
        k           = int(np.argmax(gaps) + 1)
        return max(1, min(k, max_k))

    def cluster_stats(
        self,
        embeddings: np.ndarray,
        labels:     np.ndarray,
        segments:   List[Tuple[float, float]],
    ) -> dict:
        stats = {}
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            embs = embeddings[mask]
            segs = [segments[i] for i, m in enumerate(mask) if m]
            dur  = sum(e - s for s, e in segs)
            if len(embs) > 1:
                normed = normalize(embs, norm="l2")
                sim    = float((normed @ normed.T).mean())
            else:
                sim = 1.0
            stats[f"SPEAKER_{lbl:02d}"] = {
                "num_segments":          int(mask.sum()),
                "total_duration_s":      round(dur, 2),
                "mean_intra_similarity": round(sim, 4),
            }
        return stats