# src/diarizer/pipeline.py
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

from .audio import AudioProcessor
from .clustering import SpeakerClusterer
from .embeddings import EmbeddingExtractor
from .vad import VADProcessor
from .utils.logging import get_logger
from .utils.checkpoints import CheckpointManager
from .asr import ASRTranscriber

logger = get_logger(__name__)


class DiarizationPipeline:
    def __init__(self, cfg: DictConfig, mode: str = "auto"):
        self.cfg        = cfg
        self.mode       = mode
        self.hf_token   = os.environ.get("HF_TOKEN", "")
        self.checkpointer = CheckpointManager(
            checkpoint_dir   = cfg.paths.checkpoints,
            keep_last_n      = 3,
            best_metric      = "der",
            higher_is_better = False,
        )
        self._pyannote  = None
        self._audio     = None
        self._vad       = None
        self._embedder  = None
        self._clusterer = None
        self._asr       = None
        self.use_asr = os.environ.get("USE_ASR", "false").lower() == "true"
        logger.info("DiarizationPipeline initialised — mode=%s", mode)

    def run(
        self,
        wav_path:     str,
        num_speakers: Optional[int] = None,
        output_dir:   Optional[str] = None,
        session_id:   Optional[str] = None,
    ) -> dict:
        session_id = session_id or str(uuid.uuid4())[:8]
        t_total    = time.perf_counter()

        logger.info("=" * 60)
        logger.info("Session %s | %s | mode=%s",
                    session_id, Path(wav_path).name, self.mode)

        # Step 0: preprocess audio
        audio_proc = self._get_audio_processor()
        waveform, sr, audio_meta = audio_proc.process(wav_path)
        duration_s = audio_meta["duration_s"]

        for w in audio_meta["warnings"]:
            logger.warning("[%s] %s", session_id, w)

        # Step 1: diarize
        if self.mode == "auto":
            segments = self._run_pyannote(wav_path, num_speakers)
        else:
            segments = self._run_manual(wav_path, num_speakers)

        # Step 2: transcribe if ASR is enabled
        if self.use_asr:
            asr = self._get_asr()
            segments = asr.transcribe_and_align(wav_path, segments)
            filled = sum(1 for s in segments if s["text"].strip())
            logger.info("ASR: %d/%d segments have text", filled, len(segments))
        
        # Step 3: write outputs
        out_dir = Path(output_dir or self.cfg.paths.outputs) / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        output_paths = self._write_outputs(segments, session_id, out_dir)

            
        elapsed  = time.perf_counter() - t_total
        
        speakers = set(s["speaker"] for s in segments)

        result = {
            "session_id":        session_id,
            "filename":          Path(wav_path).name,
            "audio_duration_s":  duration_s,
            "num_speakers":      len(speakers),
            "speakers":          sorted(speakers),
            "num_segments":      len(segments),
            "segments":          segments,
            "processing_time_s": round(elapsed, 3),
            "rt_factor":         round(elapsed / duration_s, 3) if duration_s > 0 else None,
            "output_paths":      output_paths,
            "audio_meta":        audio_meta,
        }

        logger.info("Session %s done | %d speakers | %d segments | %.2fs",
                    session_id, len(speakers), len(segments), elapsed)
        return result

    def _run_pyannote(self, wav_path: str, num_speakers: Optional[int]) -> List[dict]:
        import torch
        from pyannote.audio import Pipeline

        if self._pyannote is None:
            logger.info("Loading pyannote pipeline: %s", self.cfg.pipeline.model)
            t0 = time.perf_counter()

            # Set token via environment so pyannote picks it up automatically
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
            os.environ["HF_TOKEN"] = self.hf_token

            self._pyannote = Pipeline.from_pretrained(
                self.cfg.pipeline.model,
            )
            if torch.cuda.is_available() and self.cfg.pipeline.use_gpu:
                self._pyannote = self._pyannote.to(torch.device("cuda"))
            logger.info("Pipeline loaded in %.2fs", time.perf_counter() - t0)
            # Override VAD thresholds from our config
            self._pyannote._segmentation.onset            = self.cfg.vad.onset
            self._pyannote._segmentation.offset           = self.cfg.vad.offset
            self._pyannote._segmentation.min_duration_on  = self.cfg.vad.min_duration_on
            self._pyannote._segmentation.min_duration_off = self.cfg.vad.min_duration_off

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        diarization = self._pyannote(wav_path, **kwargs)
        min_dur = self.cfg.vad.min_duration_on
        return [
            {
                "speaker":  speaker,
                "start":    round(turn.start, 3),
                "end":      round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3),
                "text":     "",
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
            if (turn.end - turn.start) >= min_dur
        ]


    def _run_manual(self, wav_path: str, num_speakers: Optional[int]) -> List[dict]:
        vad      = self._get_vad()
        embedder = self._get_embedder()
        cluster  = self._get_clusterer()

        logger.info("[manual] Stage 1/3 — VAD")
        speech_segs = vad.detect(wav_path)
        if not speech_segs:
            return []

        logger.info("[manual] Stage 2/3 — Embeddings")
        embeddings, valid_segs = embedder.extract(wav_path, speech_segs)

        self.checkpointer.save(
            step         = int(time.time()),
            metric_value = 0.0,
            notes        = f"Embeddings for {Path(wav_path).name}",
        )

        logger.info("[manual] Stage 3/3 — Clustering")
        labels = cluster.cluster(embeddings, num_speakers=num_speakers)
        stats  = cluster.cluster_stats(embeddings, labels, valid_segs)
        logger.info("Cluster stats: %s", stats)

        return [
            {
                "speaker":  f"SPEAKER_{int(label):02d}",
                "start":    round(start, 3),
                "end":      round(end, 3),
                "duration": round(end - start, 3),
                "text":     "",
            }
            for (start, end), label in zip(valid_segs, labels)
        ]

    def _write_outputs(self, segments, session_id, out_dir) -> dict:
        paths = {}
        if self.cfg.output.write_rttm:
            p = out_dir / f"{session_id}.rttm"
            self._write_rttm(segments, session_id, p)
            paths["rttm"] = str(p)
        if self.cfg.output.write_json:
            p = out_dir / f"{session_id}.json"
            self._write_json(segments, session_id, p)
            paths["json"] = str(p)
        if self.cfg.output.write_srt:
            p = out_dir / f"{session_id}.srt"
            self._write_srt(segments, p)
            paths["srt"] = str(p)
        return paths

    @staticmethod
    def _write_rttm(segments, audio_id, path):
        with open(path, "w") as f:
            for s in segments:
                f.write(f"SPEAKER {audio_id} 1 "
                        f"{s['start']:.3f} {s['duration']:.3f} "
                        f"<NA> <NA> {s['speaker']} <NA> <NA>\n")

    @staticmethod
    def _write_json(segments, session_id, path):
        with open(path, "w") as f:
            json.dump({"session_id": session_id, "segments": segments}, f, indent=2)

    @staticmethod
    def _write_srt(segments, path):
        def fmt(s):
            h  = int(s // 3600)
            m  = int((s % 3600) // 60)
            sc = int(s % 60)
            ms = int((s - int(s)) * 1000)
            return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"
        with open(path, "w") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n")
                f.write(f"[{seg['speaker']}]: {seg.get('text', '')}\n\n")

    def _get_audio_processor(self):
        if self._audio is None:
            self._audio = AudioProcessor(self.cfg.audio)
        return self._audio

    def _get_vad(self):
        if self._vad is None:
            self._vad = VADProcessor(self.cfg.vad, self.hf_token)
        return self._vad

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingExtractor(
                self.cfg.embedding, self.cfg.segmentation, self.hf_token
            )
        return self._embedder

    def _get_clusterer(self):
        if self._clusterer is None:
            self._clusterer = SpeakerClusterer(self.cfg.clustering)
        return self._clusterer
    
    def _get_asr(self) -> ASRTranscriber:
        if self._asr is None:
            self._asr = ASRTranscriber(model_size="medium", device="cpu")
        return self._asr