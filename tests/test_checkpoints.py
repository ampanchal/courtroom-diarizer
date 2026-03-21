# tests/test_checkpoints.py
import pytest


@pytest.fixture
def manager(tmp_path):
    from src.diarizer.utils.checkpoints import CheckpointManager
    return CheckpointManager(
        checkpoint_dir   = str(tmp_path / "ckpts"),
        keep_last_n      = 3,
        best_metric      = "der",
        higher_is_better = False,
    )


class TestCheckpointManager:

    def test_save_creates_directory(self, manager):
        path = manager.save(step=10, metric_value=0.25)
        assert path.exists()
        assert (path / "meta.json").exists()

    def test_meta_fields(self, manager):
        manager.save(step=10, metric_value=0.20, notes="first")
        ckpts = manager.list_checkpoints()
        assert len(ckpts) == 1
        assert ckpts[0]["step"] == 10
        assert ckpts[0]["metric_value"] == 0.20
        assert ckpts[0]["notes"] == "first"

    def test_best_tracks_minimum(self, manager):
        manager.save(step=10, metric_value=0.30)
        manager.save(step=20, metric_value=0.18)
        manager.save(step=30, metric_value=0.22)
        assert manager.get_best_metric() == pytest.approx(0.18)

    def test_load_best(self, manager):
        manager.save(step=10, metric_value=0.35)
        manager.save(step=20, metric_value=0.22)
        result = manager.load_best()
        assert result["meta"]["metric_value"] == pytest.approx(0.22)

    def test_load_latest(self, manager):
        manager.save(step=10, metric_value=0.35)
        manager.save(step=20, metric_value=0.28)
        result = manager.load_latest()
        assert result["meta"]["step"] == 20

    def test_no_checkpoints_returns_none(self, manager):
        assert manager.load_best()   is None
        assert manager.load_latest() is None