# tests/conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def cfg():
    base  = OmegaConf.load("configs/base.yaml")
    model = OmegaConf.load("configs/model.yaml")
    return OmegaConf.merge(base, model)


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory):
    tmp  = tmp_path_factory.mktemp("audio")
    path = tmp / "test_hearing.wav"
    sr   = 16000
    t    = np.linspace(0, 10.0, 16000 * 10, endpoint=False)
    seg_a   = 0.3 * np.sin(2 * np.pi * 150 * t[:64000])
    silence = np.zeros(8000)
    seg_b   = 0.3 * np.sin(2 * np.pi * 250 * t[:80000])
    audio   = np.concatenate([seg_a, silence, seg_b]).astype(np.float32)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path)


@pytest.fixture(scope="session")
def sample_embeddings():
    np.random.seed(42)
    a = np.random.randn(5, 192) + 3.0
    b = np.random.randn(5, 192) - 3.0
    emb = np.vstack([a, b]).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + 1e-8)


@pytest.fixture(scope="session")
def sample_segments():
    return [(i * 1.5, i * 1.5 + 1.5) for i in range(10)]