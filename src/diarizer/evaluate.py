# src/diarizer/evaluate.py
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig

from .utils.logging import get_logger

logger = get_logger(__name__)


class DEREvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger.debug("DEREvaluator — collar=%.2fs", cfg.collar)

    def evaluate(
        self,
        ref_rttm: str,
        hyp_rttm: str,
        csv_path: Optional[str] = None,
    ) -> dict:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        ref_all = self._load_rttm(ref_rttm)
        hyp_all = self._load_rttm(hyp_rttm)

        metric = DiarizationErrorRate(
            collar       = self.cfg.collar,
            skip_overlap = self.cfg.skip_overlap,
        )
        per_file = []
        totals   = {
            "missed detection": 0.0,
            "false alarm":      0.0,
            "confusion":        0.0,
            "total":            0.0,
        }

        for file_id in sorted(set(ref_all) | set(hyp_all)):
            if file_id not in ref_all:
                logger.warning("'%s' in hyp but not in ref — skipping", file_id)
                continue
            ref  = ref_all[file_id]
            hyp  = hyp_all.get(file_id, Annotation(uri=file_id))
            comp = metric(ref, hyp, detailed=True)
            tot  = comp.get("total", 0.0)

            def rate(k):
                return comp.get(k, 0.0) / tot if tot > 0 else 0.0

            per_file.append({
                "file_id":        file_id,
                "DER":            abs(metric),
                "miss":           rate("missed detection"),
                "false_alarm":    rate("false alarm"),
                "confusion":      rate("confusion"),
                "total_speech_s": round(tot, 2),
            })
            for k in totals:
                totals[k] += comp.get(k, 0.0)

        total_der = abs(metric)
        total_ref = totals["total"]

        def pct(k):
            return totals[k] / total_ref if total_ref > 0 else 0.0

        report = {
            "total_der":      round(total_der, 4),
            "miss_rate":      round(pct("missed detection"), 4),
            "fa_rate":        round(pct("false alarm"), 4),
            "confusion_rate": round(pct("confusion"), 4),
            "total_speech_s": round(total_ref, 2),
            "collar_s":       self.cfg.collar,
            "num_files":      len(per_file),
            "per_file":       per_file,
            "quality_band":   self._quality_band(total_der),
            "diagnosis":      self._diagnose(
                pct("missed detection"),
                pct("false alarm"),
                pct("confusion"),
            ),
        }

        if total_der > self.cfg.der_threshold:
            logger.warning("DER %.2f%% exceeds threshold %.2f%%",
                           total_der * 100, self.cfg.der_threshold * 100)

        logger.info("DER=%.2f%%  Miss=%.2f%%  FA=%.2f%%  Conf=%.2f%%  [%s]",
                    total_der * 100,
                    pct("missed detection") * 100,
                    pct("false alarm") * 100,
                    pct("confusion") * 100,
                    report["quality_band"])

        if csv_path:
            self._write_csv(per_file, csv_path)

        return report

    @staticmethod
    def _load_rttm(path: str) -> Dict:
        from pyannote.core import Annotation, Segment

        annotations = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue
                parts = line.split()
                if len(parts) < 9 or parts[0] != "SPEAKER":
                    continue
                file_id  = parts[1]
                onset    = float(parts[3])
                dur      = float(parts[4])
                speaker  = parts[7]
                if file_id not in annotations:
                    annotations[file_id] = Annotation(uri=file_id)
                seg = Segment(onset, onset + dur)
                annotations[file_id][seg, speaker] = speaker

        if not annotations:
            raise ValueError(f"No SPEAKER records found in: {path}")
        return annotations

    @staticmethod
    def _quality_band(der: float) -> str:
        if der < 0.10: return "EXCELLENT (<10%)"
        if der < 0.20: return "GOOD (<20%)"
        if der < 0.35: return "FAIR (<35%)"
        return "POOR (>35%)"

    @staticmethod
    def _diagnose(miss: float, fa: float, conf: float) -> str:
        dominant = max(
            [("Miss", miss), ("FA", fa), ("Confusion", conf)],
            key=lambda x: x[1],
        )
        tips = {
            "Miss":      "Lower VAD onset — soft voices being cut.",
            "FA":        "Raise VAD onset — noise detected as speech.",
            "Confusion": "Tune distance_threshold or fine-tune embeddings.",
        }
        return tips[dominant[0]]

    @staticmethod
    def _write_csv(per_file: list, path: str) -> None:
        fields = ["file_id", "DER", "miss", "false_alarm",
                  "confusion", "total_speech_s"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_file)