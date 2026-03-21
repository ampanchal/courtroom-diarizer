# tests/test_clustering.py
import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def clusterer(cfg):
    from src.diarizer.clustering import SpeakerClusterer
    return SpeakerClusterer(cfg.clustering)


class TestAHC:

    def test_two_clusters_detected(self, clusterer, sample_embeddings):
        labels = clusterer.cluster(sample_embeddings, num_speakers=2)
        assert len(set(labels)) == 2

    def test_fixed_k_respected(self, clusterer, sample_embeddings):
        for k in [2, 3, 4]:
            labels = clusterer.cluster(sample_embeddings, num_speakers=k)
            assert len(set(labels)) == k

    def test_single_segment(self, clusterer):
        emb = np.random.randn(1, 192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        assert list(clusterer.cluster(emb)) == [0]

    def test_empty_raises(self, clusterer):
        with pytest.raises(ValueError):
            clusterer.cluster(np.empty((0, 192)))


class TestClusterStats:

    def test_counts_sum_to_total(self, clusterer, sample_embeddings, sample_segments):
        labels = clusterer.cluster(sample_embeddings, num_speakers=2)
        stats  = clusterer.cluster_stats(sample_embeddings, labels, sample_segments)
        total  = sum(v["num_segments"] for v in stats.values())
        assert total == len(sample_embeddings)

    def test_similarity_in_range(self, clusterer, sample_embeddings, sample_segments):
        labels = clusterer.cluster(sample_embeddings, num_speakers=2)
        stats  = clusterer.cluster_stats(sample_embeddings, labels, sample_segments)
        for info in stats.values():
            assert 0.0 <= info["mean_intra_similarity"] <= 1.0 + 1e-6