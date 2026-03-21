# src/diarizer/utils/logging.py
import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: str = "configs/logging.yaml",
    log_dir: str = "logs",
    level: Optional[str] = None,
) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                handler["filename"] = str(
                    Path(log_dir) / Path(handler["filename"]).name
                )
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    effective_level = level or os.environ.get("LOG_LEVEL", "").upper()
    if effective_level:
        logging.getLogger("diarizer").setLevel(effective_level)

    logger = logging.getLogger("diarizer")
    logger.info("Logging initialised — writing to %s/", log_dir)
    return logger


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("diarizer"):
        name = f"diarizer.{name}"
    return logging.getLogger(name)