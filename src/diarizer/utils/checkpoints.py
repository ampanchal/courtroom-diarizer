# src/diarizer/utils/checkpoints.py
from importlib.resources import path
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointMeta:
    checkpoint_id:   str
    step:            int
    epoch:           Optional[int]
    metric_name:     str
    metric_value:    float
    timestamp:       str
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    notes:           str = ""


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir:   str = "checkpoints",
        keep_last_n:      int = 3,
        best_metric:      str = "der",
        higher_is_better: bool = False,
    ):
        self.checkpoint_dir   = Path(checkpoint_dir)
        self.keep_last_n      = keep_last_n
        self.best_metric      = best_metric
        self.higher_is_better = higher_is_better
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("CheckpointManager initialised at %s", checkpoint_dir)

    def save(
        self,
        step:            int,
        metric_value:    float,
        state:           Optional[Dict] = None,
        epoch:           Optional[int] = None,
        config_snapshot: Optional[Dict] = None,
        notes:           str = "",
    ) -> Path:
        checkpoint_id = f"checkpoint_{step:06d}"
        ckpt_path     = self.checkpoint_dir / checkpoint_id
        ckpt_path.mkdir(exist_ok=True)

        meta = CheckpointMeta(
            checkpoint_id   = checkpoint_id,
            step            = step,
            epoch           = epoch,
            metric_name     = self.best_metric,
            metric_value    = metric_value,
            timestamp       = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            config_snapshot = config_snapshot or {},
            notes           = notes,
        )

        with open(ckpt_path / "meta.json", "w") as f:
            json.dump(asdict(meta), f, indent=2)

        if state is not None:
            import torch
            torch.save(state, ckpt_path / "state.pt")

        logger.info("Checkpoint saved: %s  |  %s=%.4f",
                    checkpoint_id, self.best_metric, metric_value)
        self._update_best(ckpt_path, metric_value)
        self._prune()
        return ckpt_path

    def load_best(self) -> Optional[Dict]:
        best_path = self.checkpoint_dir / "best"
        if not best_path.exists():
            return None
        return self._load_from(best_path.resolve())

    def load_latest(self) -> Optional[Dict]:
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        return self._load_from(checkpoints[-1])

    def list_checkpoints(self) -> List[Dict]:
        metas = []
        for ckpt_dir in sorted(self._list_checkpoints(), reverse=True):
            meta_path = ckpt_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    metas.append(json.load(f))
        return metas

    def get_best_metric(self) -> Optional[float]:
        best_path = self.checkpoint_dir / "best" / "meta.json"
        if not best_path.exists():
            return None
        with open(best_path) as f:
            return json.load(f)["metric_value"]

    def _list_checkpoints(self) -> List[Path]:
        return sorted(
            p for p in self.checkpoint_dir.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint_")
        )

    def _update_best(self, ckpt_path: Path, metric_value: float) -> None:
        best_link    = self.checkpoint_dir / "best"
        current_best = self.get_best_metric()
        is_better = (
            current_best is None
            or (self.higher_is_better and metric_value > current_best)
            or (not self.higher_is_better and metric_value < current_best)
        )
        if is_better:
            if best_link.exists() or best_link.is_symlink():
                if best_link.is_symlink() or best_link.is_file():
                    best_link.unlink()
                else:
                    shutil.rmtree(best_link)
            shutil.copytree(ckpt_path, best_link)
            logger.info("New best checkpoint: %s=%.4f", self.best_metric, metric_value)

    def _load_from(self, path: Path) -> Dict:
        meta_path  = path / "meta.json"
        state_path = path / "state.pt"
        result = {}
        if meta_path.exists():
            with open(meta_path) as f:
                result["meta"] = json.load(f)
        if state_path.exists():
            import torch
            result["state"] = torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
        logger.info("Checkpoint loaded from %s", path)
        return result

    def _prune(self) -> None:
        checkpoints = self._list_checkpoints()
        to_delete   = checkpoints[: max(0, len(checkpoints) - self.keep_last_n)]
        for ckpt in to_delete:
            shutil.rmtree(ckpt)
            logger.debug("Pruned old checkpoint: %s", ckpt.name)