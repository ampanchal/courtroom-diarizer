# tests/test_audio.py
import numpy as np
import pytest
import soundfile as sf


class TestAudioProcessor:

    def test_load_and_normalize(self, cfg, sample_wav):
        from src.diarizer.audio import AudioProcessor
        import torch
        proc = AudioProcessor(cfg.audio)
        waveform, sr, meta = proc.process(sample_wav)
        assert sr == 16000
        assert waveform.shape[0] == 1
        assert meta["duration_s"] > 0

    def test_output_is_mono(self, cfg, sample_wav):
        from src.diarizer.audio import AudioProcessor
        proc = AudioProcessor(cfg.audio)
        waveform, sr, _ = proc.process(sample_wav)
        assert waveform.shape[0] == 1

    def test_no_clipping(self, cfg, sample_wav):
        from src.diarizer.audio import AudioProcessor
        import torch
        proc = AudioProcessor(cfg.audio)
        waveform, _, meta = proc.process(sample_wav)
        assert waveform.abs().max() <= 1.0
        assert meta["clip_pct"] <= cfg.audio.max_clip_pct

    def test_file_hash_present(self, cfg, sample_wav):
        from src.diarizer.audio import AudioProcessor
        proc = AudioProcessor(cfg.audio)
        _, _, meta = proc.process(sample_wav)
        assert "file_hash" in meta
        assert len(meta["file_hash"]) == 16

    def test_missing_file_raises(self, cfg):
        from src.diarizer.audio import AudioProcessor
        proc = AudioProcessor(cfg.audio)
        with pytest.raises(FileNotFoundError):
            proc.process("does_not_exist.wav")

    def test_short_audio_warning(self, cfg, tmp_path):
        from src.diarizer.audio import AudioProcessor
        p = tmp_path / "short.wav"
        sf.write(str(p), np.zeros(4800, dtype=np.float32), 16000)
        proc = AudioProcessor(cfg.audio)
        _, _, meta = proc.process(str(p))
        assert any("short" in w.lower() for w in meta["warnings"])