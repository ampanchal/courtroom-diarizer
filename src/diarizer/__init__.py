# src/diarizer/__init__.py
from .pipeline import DiarizationPipeline
from .audio import AudioProcessor
from .vad import VADProcessor
from .embeddings import EmbeddingExtractor
from .clustering import SpeakerClusterer
from .utils.logging import setup_logging, get_logger
from .utils.checkpoints import CheckpointManager

__version__ = "1.0.0"

__all__ = [
    "DiarizationPipeline",
    "AudioProcessor",
    "VADProcessor",
    "EmbeddingExtractor",
    "SpeakerClusterer",
    "setup_logging",
    "get_logger",
    "CheckpointManager",
]