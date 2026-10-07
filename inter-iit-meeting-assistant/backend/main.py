"""FastAPI server: upload -> Whisper -> refinement LLM -> extraction LLM -> exports.

Run from this folder:   uvicorn main:app --reload
Then open http://127.0.0.1:8000
"""

from dotenv import load_dotenv

load_dotenv()  # must run before the processor modules read their env vars

import copy
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

import audio_processor
import text_processor
from errors import EmptyFileError, PipelineError, UnsupportedFileError
from exporter import to_pdf, to_text

logger = logging.getLogger("uvicorn.error")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
JOB_TTL_SECONDS = 2 * 60 * 60

def _models() -> dict:
    """Model names for display; reflects any fallback model that was actually used."""
    return {
        "speech_to_text": f"OpenAI Whisper via faster-whisper ({audio_processor.WHISPER_MODEL_NAME}, local)",
        "refinement": text_processor.active_model(text_processor.REFINE_MODEL),
        "extraction": text_processor.active_model(text_processor.EXTRACT_MODEL),
    }

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:  # load Whisper once at startup so the first upload isn't slow
        audio_processor._get_model()
        logger.info("Whisper model '%s' loaded.", audio_processor.WHISPER_MODEL_NAME)
    except Exception:
        logger.exception("Could not preload Whisper; uploads will fail until this is fixed")
    yield


app = FastAPI(title="Meeting Assistant", lifespan=lifespan)


# --------------------------------------------------------------- job helpers
def _update(job_id: str, **fields):
    with LOCK:
        JOBS[job_id].update(fields)


def _set_stage(job_id: str, stage: str, state: str, message: str | None = None):
    with LOCK:
        JOBS[job_id]["stages"][stage] = state
        if message is not None:
            JOBS[job_id]["message"] = message


def _purge_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with LOCK:
        for jid in [j for j, v in JOBS.items() if v["created"] < cutoff]:
            del JOBS[jid]


def _parse_glossary(raw: str) -> list[str]:
    terms = [t.strip() for t in re.split(r"[,\n;]", raw or "") if t.strip()]
    return terms[:100]


def _remove_file(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


def run_job(job_id: str, path: str, glossary: list[str]):
    """The sequential pipeline. Each stage's output is stored as soon as it exists."""
    stage = "transcribe"
    try:
        _update(job_id, status="running")

        _set_stage(job_id, "transcribe", "running", "Transcribing audio - this can take a few minutes for long recordings...")
        t0 = time.perf_counter()
        raw = audio_processor.transcribe_audio(path)
        logger.info("job %s: transcribe took %.1fs", job_id, time.perf_counter() - t0)
        _update(job_id, raw_transcript=raw)
        _set_stage(job_id, "transcribe", "done")

        stage = "refine"
        _set_stage(job_id, "refine", "running", "Refining domain terminology...")
        t0 = time.perf_counter()
        refined = text_processor.refine_transcript(raw, glossary)
        logger.info("job %s: refine took %.1fs", job_id, time.perf_counter() - t0)
        _update(job_id, refined_transcript=refined)
        _set_stage(job_id, "refine", "done")

        stage = "extract"
        _set_stage(job_id, "extract", "running", "Generating minutes, decisions and action items...")
        t0 = time.perf_counter()
        doc = text_processor.extract_documentation(refined)
        logger.info("job %s: extract took %.1fs", job_id, time.perf_counter() - t0)
        _set_stage(job_id, "extract", "done", "Done.")
        _update(job_id, record=doc, status="done",
                finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    except PipelineError as e:
        logger.error("job %s failed in %s: %s", job_id, stage, e.message, exc_info=e.__cause__ or e)
        _set_stage(job_id, stage, "error")
        _update(job_id, status="error", error=e.message)
    except Exception:  # never leave a job hanging
        logger.exception("job %s failed in %s", job_id, stage)
        _set_stage(job_id, stage, "error")
        _update(job_id, status="error", error="Something went wrong while processing the recording.")
    finally:
        _remove_file(path)


def _public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "created"}


def _build_record(job: dict) -> dict:
    r = job["record"]
    return {
        "source_file": job["filename"],
        "generated_at": job["finished_at"],
        "models": _models(),
        "raw_transcript": job["raw_transcript"],
        "refined_transcript": job["refined_transcript"],
        "summary": r["summary"],
        "minutes": r["minutes"],
        "key_decisions": r["key_decisions"],
        "action_items": r["action_items"],
    }


# --------------------------------------------------------------- API routes
@app.get("/api/info")
def info():
    return {"models": _models(), "supported_extensions": sorted(audio_processor.SUPPORTED_EXTENSIONS),
            "max_upload_mb": MAX_UPLOAD_MB}


@app.post("/api/jobs", status_code=202)
def create_job(background: BackgroundTasks, file: UploadFile = File(...), glossary: str = Form("")):
    _purge_old_jobs()
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    tmp_name = None
    try:
        if suffix not in audio_processor.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(
                f"Unsupported file type '{suffix or 'unknown'}'. Please upload one of: "
                + ", ".join(sorted(audio_processor.SUPPORTED_EXTENSIONS))
            )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_name = tmp.name
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with tmp:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise UnsupportedFileError(f"The file is larger than the {MAX_UPLOAD_MB} MB limit.")
                tmp.write(chunk)
        if size == 0:
            raise EmptyFileError("The uploaded file is empty.")
        audio_processor.validate_audio_file(tmp_name)
    except PipelineError as e:
        if tmp_name:
            _remove_file(tmp_name)
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except Exception:
        if tmp_name:
            _remove_file(tmp_name)
        logger.exception("upload failed")
        raise HTTPException(status_code=500, detail="The upload could not be saved on the server.")

    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {
            "id": job_id, "filename": filename, "status": "queued", "created": time.time(),
            "stages": {"transcribe": "pending", "refine": "pending", "extract": "pending"},
            "message": "Queued...", "raw_transcript": None, "refined_transcript": None,
            "record": None, "error": None, "finished_at": None,
        }
    background.add_task(run_job, job_id, tmp_name, _parse_glossary(glossary))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found. It may have expired - please upload again.")
        return _public_job(copy.deepcopy(job))


@app.get("/api/jobs/{job_id}/export/{fmt}")
def export_job(job_id: str, fmt: str):
    with LOCK:
        job = JOBS.get(job_id)
        job = copy.deepcopy(job) if job else None
    if not job:
        raise HTTPException(404, "Job not found. It may have expired - please upload again.")
    if job["status"] != "done":
        raise HTTPException(409, "The meeting record is not ready yet.")

    rec = _build_record(job)
    stem = f"meeting-record-{Path(job['filename']).stem or job_id}"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    if fmt == "json":
        body, mime = json.dumps(rec, indent=2, ensure_ascii=False).encode("utf-8"), "application/json"
    elif fmt == "txt":
        body, mime = to_text(rec).encode("utf-8"), "text/plain; charset=utf-8"
    elif fmt == "pdf":
        body, mime = to_pdf(rec), "application/pdf"
    else:
        raise HTTPException(400, "Unknown export format. Use json, txt or pdf.")
    return Response(body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"'})


# Frontend (must be mounted last so it doesn't shadow /api)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
