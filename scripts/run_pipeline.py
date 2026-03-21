# scripts/run_pipeline.py
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from src.diarizer import DiarizationPipeline
from src.diarizer.utils.logging import setup_logging


def parse_args():
    p = argparse.ArgumentParser(description="Courtroom Speaker Diarization")
    p.add_argument("audio",          nargs="+",  help="Input audio file(s)")
    p.add_argument("--config",       default="configs/base.yaml")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--mode",         choices=["auto", "manual"], default="auto")
    p.add_argument("--num-speakers", type=int, default=None)
    p.add_argument("--output-dir",   default="outputs")
    p.add_argument("--log-level",    default="INFO")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logging("configs/logging.yaml",
                           log_dir="logs", level=args.log_level)

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.load(args.model_config),
    )

    logger.info("Mode: %s | Speakers: %s",
                args.mode, args.num_speakers or "auto")

    pipeline = DiarizationPipeline(cfg, mode=args.mode)

    for audio_path in args.audio:
        if not Path(audio_path).exists():
            logger.error("File not found: %s — skipping", audio_path)
            continue

        session_id = Path(audio_path).stem
        result     = pipeline.run(
            wav_path     = audio_path,
            num_speakers = args.num_speakers,
            output_dir   = args.output_dir,
            session_id   = session_id,
        )

        print(f"\n{'─'*54}")
        print(f"  Session  : {result['session_id']}")
        print(f"  File     : {result['filename']}")
        print(f"  Duration : {result['audio_duration_s']:.1f}s")
        print(f"  Speakers : {result['num_speakers']} "
              f"→ {', '.join(result['speakers'])}")
        print(f"  Segments : {result['num_segments']}")
        print(f"  Time     : {result['processing_time_s']:.2f}s "
              f"({result['rt_factor']:.2f}x RT)")
        print(f"\n  Outputs  : {args.output_dir}/{session_id}/")
        print(f"{'─'*54}\n")

        # Print first 5 segments as preview
        print("  First segments:")
        for seg in result["segments"][:5]:
            print(f"    {seg['start']:>7.2f}s → {seg['end']:>7.2f}s  "
                  f"[{seg['speaker']}]")
        if result["num_segments"] > 5:
            print(f"    ... and {result['num_segments'] - 5} more segments")


if __name__ == "__main__":
    main()