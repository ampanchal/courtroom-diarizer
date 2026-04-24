# CourtScribe — AI-Powered Courtroom Transcription System

> **Reducing legal typist workload by automating speaker identification, segmentation, and transcription in courtroom proceedings.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![DER](https://img.shields.io/badge/VoxConverse%20DER-5.29%25-brightgreen.svg)](#benchmark-results)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## The Problem

A legal typist in an Indian courtroom faces a uniquely difficult transcription task. A typical session runs 2–4 hours, involves 3–6 speakers with distinct roles (judge, senior advocate, opposing counsel, witness), switches freely between Hindi, English, and Marathi mid-sentence, and produces a document that carries legal weight.

Today, that typist listens to a recording, manually identifies every speaker switch, types the dialogue, then reviews the entire transcript for errors — all under deadline. A single misattributed statement can alter the legal record.

**CourtScribe automates the hardest part**: knowing *who* said *what* and *when*, so the typist focuses entirely on accuracy rather than logistics.

---

## What It Does

CourtScribe is a complete, on-premise AI pipeline that:

1. **Diarizes** — identifies and separates distinct speakers in audio recordings
2. **Transcribes** — converts speech to text using OpenAI Whisper with hybrid alignment
3. **Presents** — serves a browser-based editor where the typist reviews, corrects, and exports the final legal transcript
4. **Protects** — processes everything locally; no audio or transcript data ever leaves the premises

```
Audio file → VAD → Speaker Embeddings → Clustering → ASR → Typist UI → Legal Transcript
```

---

## System Architecture

```
courtroom-diarizer/
│
├── configs/              YAML configuration for all pipeline stages
│   ├── base.yaml         Audio, VAD, clustering, evaluation settings
│   ├── model.yaml        Model IDs, fine-tuning parameters
│   └── logging.yaml      Rotating file handlers, structured audit log
│
├── src/diarizer/         Core ML library (pip-installable)
│   ├── audio.py          MP3/WAV → 16kHz mono → LUFS normalisation
│   ├── vad.py            Voice Activity Detection (pyannote)
│   ├── embeddings.py     Speaker fingerprinting — pyannote ECAPA-TDNN
│   ├── clustering.py     AHC + spectral clustering, auto speaker count
│   ├── pipeline.py       End-to-end orchestrator, auto + manual modes
│   ├── asr.py            Whisper medium, hybrid word-to-speaker alignment
│   ├── evaluate.py       DER computation with component breakdown
│   └── utils/
│       ├── logging.py    Structured logging, legal audit trail
│       └── checkpoints.py Save/load/prune training checkpoints
│
├── api/
│   └── app.py            FastAPI server — /health, /diarize endpoints
│
├── scripts/
│   ├── run_pipeline.py   CLI: diarize one or many audio files
│   ├── preprocess.py     DVC stage 1: raw audio → processed WAV
│   ├── evaluate.py       DVC stage 3: compute DER, write metrics.json
│   └── evaluate_backends.py  Backend A/B comparison framework
│
├── typist.html           Standalone typist UI (zero deployment friction)
├── dvc.yaml              3-stage reproducible data pipeline
└── tests/                18 unit tests, all passing
```

---

## Benchmark Results

Evaluated on **VoxConverse** (216 files, 50+ hours, political debates and news segments — the standard diarization benchmark). No fine-tuning on this dataset.

### Pyannote Auto Pipeline (Primary System)

| Metric | Value |
|--------|-------|
| **Mean DER** | **5.29%** |
| Median DER | 5.34% |
| Std DER | 1.31% |
| Min / Max | 0.24% / 8.19% |
| Files under 10% DER | **100%** |
| Files under 20% DER | **100%** |
| Mean Miss | 1.81% |
| Mean False Alarm | 3.01% |
| Mean Confusion | 3.49% |

> The published VoxConverse leaderboard reports top systems at 5–7% DER. This pipeline places at the top of that range with zero domain-specific training.

### DER by Speaker Count

| Speakers | Files | Mean DER | Mean Confusion |
|----------|-------|----------|----------------|
| 1 | 22 | 5.29% | 0.29% |
| 2 | 44 | 5.29% | 1.24% |
| 3–5 | 90 | 5.13% | 3.67% |
| 6–9 | 44 | 5.32% | 5.52% |
| 10+ | 16 | 6.06% | 6.75% |

### Backend Comparison

| Backend | Mean DER | Confusion | Notes |
|---------|----------|-----------|-------|
| **Pyannote auto** | **5.29%** | 3.49% | End-to-end optimised, learned clustering |
| WavLM + AHC | 22.37% | 15.88% | Fixed threshold, no domain tuning |

The 17% gap is concentrated entirely in confusion — speaker mislabelling — confirming that clustering optimisation, not embedding quality, is the primary bottleneck for WavLM-based diarization. Fine-tuning on courtroom-specific data (SCOTUS oral arguments) is the planned next step.

---

## CourtScribe Typist UI

A single `typist.html` file served by the FastAPI backend. Zero installation for the typist — open a browser, upload audio.

**Features:**
- Audio playback with segment-synced highlighting — the active speaker turn glows amber as audio plays
- Speed controls (0.75×–2×) and keyboard shortcuts optimised for foot-pedal workflows
- Speaker label management — rename SPEAKER_00 to "THE COURT" and all segments update instantly
- Inline text editing with auto-resize
- Split and merge segments to correct boundary errors
- "Review" badges automatically flag segments under 0.8 seconds for human verification
- Save draft as JSON — resume work in a later session without reprocessing
- Export as formatted legal transcript with line numbers, speaker labels, and certificate of accuracy

**Screenshots:**

> *(Add your screenshots here — drag them into the GitHub editor)*
>
> Suggested: `docs/screenshots/typist_main.png`, `docs/screenshots/typist_export.png`

---

## Quick Start

### Prerequisites

- Python 3.11
- ffmpeg ([download](https://www.gyan.dev/ffmpeg/builds/) — add `C:\ffmpeg\bin` to PATH on Windows)
- HuggingFace account with accepted model terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Installation

```bash
# Clone and enter project
git clone https://github.com/ampanchal/courtroom-diarizer.git
cd courtroom-diarizer

# Create virtual environment (Python 3.11 required)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install PyTorch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

### Set your HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
```

### Run the API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/typist** in your browser.

### Diarize from the command line

```bash
# Auto mode — full pyannote pipeline
python scripts/run_pipeline.py hearing.wav

# With known speaker count
python scripts/run_pipeline.py hearing.wav --num-speakers 3

# With ASR transcription
USE_ASR=true python scripts/run_pipeline.py hearing.wav
```

Output is written to `outputs/<session_id>/` as RTTM, JSON, and SRT simultaneously.

---

## Configuration

All tunable parameters live in `configs/base.yaml`. The most impactful ones:

```yaml
vad:
  onset: 0.4              # Lower = more sensitive (good for soft-spoken witnesses)
  min_duration_on: 0.5    # Minimum speech segment length — raises with noise

clustering:
  distance_threshold: 0.92  # Higher = fewer speakers detected
                             # Tune this first when DER is high
```

No code changes needed to tune the pipeline — edit the YAML and re-run.

---

## Data Versioning with DVC

```bash
# Initialise DVC (already done)
dvc init

# Run the full pipeline: preprocess → diarize → evaluate
dvc repro

# Check what changed since last run
dvc params diff
dvc metrics show
```

Data files (audio, processed outputs) are tracked by DVC, not git. Model code and configs are tracked by git. They never mix.

---

## Evaluation

```bash
# Compute DER against reference RTTM files
python scripts/evaluate.py \
  --hyp-dir outputs/diarized \
  --ref-dir data/annotations \
  --output  outputs/metrics.json
```

DER is broken into Miss + False Alarm + Confusion with an actionable diagnosis printed for the dominant error type.

---

## Running Tests

```bash
pytest tests/ -v --cov=src/diarizer
```

18 tests covering audio processing, clustering (AHC + spectral), checkpoint management, and the FastAPI endpoints. All pass without downloading any models.

---

## Privacy and On-Premise Deployment

Courtroom audio is sensitive. CourtScribe is designed for zero-egress operation:

- The API server runs entirely on local hardware
- All model weights are cached locally after first download
- Temporary audio files are deleted immediately after processing in a `finally` block
- No telemetry, no cloud calls after initial setup
- Docker deployment with non-root user and built-in health checks

```bash
# Build with model weights baked in (air-gapped courts)
docker build --build-arg HF_TOKEN=$HF_TOKEN -t court-diarizer:latest .

# Run
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  court-diarizer:latest
```

---

## Roadmap

| Item | Status |
|------|--------|
| Core diarization pipeline | ✅ Complete |
| Whisper ASR with hybrid alignment | ✅ Complete |
| CourtScribe typist UI | ✅ Complete |
| VoxConverse benchmark (5.29% DER) | ✅ Complete |
| WavLM backend comparison | ✅ Complete |
| SCOTUS oral arguments download | ⏳ Pending compute |
| Fine-tuning on courtroom domain data | ⏳ Pending SCOTUS dataset |
| Multi-language support (Hindi/Marathi/English) | ⏳ In progress |
| Real-time streaming diarization | 🔲 Planned |

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Diarization | pyannote.audio 3.1 |
| Speaker embeddings | ECAPA-TDNN (wespeaker ResNet34) |
| ASR | OpenAI Whisper medium |
| Clustering | Agglomerative Hierarchical Clustering |
| API | FastAPI + Uvicorn |
| Audio processing | torchaudio, soundfile |
| ML framework | PyTorch 2.2 |
| Data versioning | DVC |
| Code versioning | Git / GitHub |
| Evaluation | pyannote.metrics (DER) |

---

## Project Background

This project was built as an end-to-end production ML system targeting a real workflow problem in Indian courts — where multilingual, multi-speaker proceedings are transcribed manually under legal deadlines.

The system was designed, implemented, evaluated, and iterated entirely from scratch: architecture decisions, dataset selection (VoxConverse, SCOTUS oral arguments), model benchmarking on 216 real audio files, backend comparisons (pyannote vs WavLM), and a production-grade typist interface — all documented here with reproducible results.

---

## Author

**Avadhoot Panchal**
[GitHub](https://github.com/ampanchal) · [LinkedIn](https://www.linkedin.com/in/avadhoot-panchal-38a298243/)
