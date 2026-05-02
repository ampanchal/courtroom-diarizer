# api/app.py
import asyncio
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import sys
from fastapi.responses import FileResponse
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.diarizer import DiarizationPipeline
from src.diarizer.utils.logging import setup_logging

logger = setup_logging("configs/logging.yaml", log_dir="logs")

cfg = OmegaConf.merge(
    OmegaConf.load("configs/base.yaml"),
    OmegaConf.load("configs/model.yaml"),
)

MAX_UPLOAD_MB   = int(os.environ.get("MAX_UPLOAD_MB", "500"))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
PIPELINE_MODE   = os.environ.get("PIPELINE_MODE", "auto")

app = FastAPI(
    title       = "Courtroom Diarization API",
    description = "On-premise speaker diarization. Audio never leaves this server.",
    version     = "1.0.0",
)

@app.get("/typist")
async def typist_ui():
    return FileResponse(
        "typist.html",
        headers={
            # Prevent stale cached UI assets from old runs.
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

pipeline: Optional[DiarizationPipeline] = None
executor = ThreadPoolExecutor(max_workers=1)


# ── Schema ────────────────────────────────────────────────────
class Segment(BaseModel):
    speaker:  str
    start:    float
    end:      float
    duration: float
    text:     str = ""

class DiarizationResponse(BaseModel):
    session_id:        str
    audio_duration_s:  float
    num_speakers:      int
    speakers:          list[str]
    segments:          list[Segment]
    processing_time_s: float
    rt_factor:         Optional[float] = None

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status:       str
    model_loaded: bool
    mode:         str
    version:      str


# ── Lifecycle ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global pipeline
    logger.info("API startup — mode=%s", PIPELINE_MODE)
    pipeline = DiarizationPipeline(cfg, mode=PIPELINE_MODE)
    logger.info("Pipeline object created — model loads on first request")


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status       = "healthy" if pipeline is not None else "loading",
        model_loaded = pipeline is not None,
        mode         = PIPELINE_MODE,
        version      = "1.0.0",
    )


@app.post("/diarize", response_model=DiarizationResponse)
async def diarize(
    file:         UploadFile = File(...),
    num_speakers: Optional[int] = Query(None, ge=1, le=20),
):
    if pipeline is None:
        raise HTTPException(503, detail="Pipeline not ready — retry in a moment.")

    content = await file.read()
    if len(content) / (1024 ** 2) > MAX_UPLOAD_MB:
        raise HTTPException(413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    session_id = str(uuid.uuid4())[:8]
    suffix     = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path   = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=f"court_{session_id}_",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        del content

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: pipeline.run(
                wav_path     = tmp_path,
                num_speakers = num_speakers,
                session_id   = session_id,
            ),
        )

        return DiarizationResponse(
            session_id        = result["session_id"],
            audio_duration_s  = result["audio_duration_s"],
            num_speakers      = result["num_speakers"],
            speakers          = result["speakers"],
            segments          = [Segment(**s) for s in result["segments"]],
            processing_time_s = result["processing_time_s"],
            rt_factor         = result["rt_factor"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[%s] Diarization failed: %s", session_id, exc, exc_info=True)
        raise HTTPException(500, detail=str(exc))
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/")
async def root():
    return {
        "service": "Courtroom Diarization API",
        "docs":    "/docs",
        "health":  "/health",
    }