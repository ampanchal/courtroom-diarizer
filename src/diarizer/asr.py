# src/diarizer/asr.py
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .utils.logging import get_logger

logger = get_logger(__name__)


class ASRTranscriber:
    """
    Whisper-based ASR using the hybrid alignment strategy:
      1. Transcribe the full audio file once for maximum context
      2. Align Whisper word-level timestamps to diarization segments
      3. Assign each word to the speaker whose time window contains it

    Supports Hindi, English, Marathi and code-switched audio.
    """

    def __init__(self, model_size: str = "medium", device: str = "cpu"):
        self.model_size = model_size
        self.device     = device
        self._model     = None
        logger.info("ASRTranscriber initialised — model=%s device=%s",
                    model_size, device)

    def _load(self) -> None:
        if self._model is not None:
            return
        import whisper
        logger.info("Loading Whisper %s ...", self.model_size)
        t0 = time.perf_counter()
        self._model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Whisper loaded in %.2fs", time.perf_counter() - t0)

    def transcribe_and_align(
        self,
        wav_path:  str,
        segments:  List[dict],
        language:  Optional[str] = None,
    ) -> List[dict]:
        """
        Transcribe full audio then align words to speaker segments.

        Args:
            wav_path: Path to 16kHz mono WAV file.
            segments: Diarization segments from pipeline.run()
                      Each has: speaker, start, end, duration, text
            language: Force language code e.g. 'hi' for Hindi.
                      None = Whisper auto-detects per segment.

        Returns:
            Same segments list with 'text' field populated.
        """
        self._load()
        import whisper

        logger.info("Transcribing %s with Whisper %s ...",
                    Path(wav_path).name, self.model_size)
        t0 = time.perf_counter()

        # Step 1: full-file transcription with word timestamps
        result = self._model.transcribe(
            wav_path,
            language        = language,
            word_timestamps = True,
            verbose         = False,
            task            = "transcribe",
            condition_on_previous_text = True,  # helps with code-switching
        )

        logger.info("Whisper transcription done in %.2fs — %d segments detected",
                    time.perf_counter() - t0, len(result["segments"]))

        # Step 2: extract all words with timestamps
        words = self._extract_words(result)
        logger.info("Extracted %d words total", len(words))

        # Step 3: align words to speaker segments
        aligned = self._align_words_to_segments(words, segments)

        filled = sum(1 for s in aligned if s["text"].strip())
        logger.info("Aligned text to %d/%d segments", filled, len(aligned))

        return aligned

    def _extract_words(self, whisper_result: dict) -> List[dict]:
        """
        Pull every word with its start/end time from Whisper output.
        Whisper nests words inside segments inside the result dict.
        """
        words = []
        for seg in whisper_result.get("segments", []):
            for w in seg.get("words", []):
                word  = w.get("word", "").strip()
                start = w.get("start", 0.0)
                end   = w.get("end",   0.0)
                if word:
                    words.append({
                        "word":  word,
                        "start": start,
                        "end":   end,
                    })
        return words

    def _align_words_to_segments(
        self,
        words:    List[dict],
        segments: List[dict],
    ) -> List[dict]:
        """
        For each word, find which diarization segment's time window
        contains its midpoint and assign it to that speaker.

        Using midpoint rather than start avoids edge cases where a
        word starts at the boundary between two speaker turns.
        """
        aligned = [dict(s) for s in segments]  # deep copy

        # Build a text bucket for each segment
        buckets = [[] for _ in aligned]

        for word in words:
            midpoint = (word["start"] + word["end"]) / 2
            # Find the segment whose window contains this midpoint
            for i, seg in enumerate(aligned):
                if (seg["start"] - 0.3) <= midpoint <= (seg["end"] + 0.3):
                    buckets[i].append(word["word"])
                    break

        # Join words into text for each segment
        for i, seg in enumerate(aligned):
            aligned[i]["text"] = " ".join(buckets[i]).strip()

        return aligned

    def detect_language(self, wav_path: str) -> str:
        """
        Detect the dominant language in the first 30 seconds.
        Returns a language code like 'hi', 'en', 'mr'.
        Useful for logging and debugging code-switched audio.
        """
        self._load()
        import whisper

        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel   = whisper.log_mel_spectrogram(audio).to(self.device)

        _, probs = self._model.detect_language(mel)
        top_lang  = max(probs, key=probs.get)
        top_prob  = probs[top_lang]

        logger.info("Detected language: %s (%.1f%%)", top_lang, top_prob * 100)
        return top_lang