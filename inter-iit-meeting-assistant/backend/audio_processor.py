"""Stage 0: speech-to-text with a local faster-whisper model (CTranslate2, int8).

Audio is decoded with the system ffmpeg (must be on PATH) into raw samples, which are passed
straight to faster-whisper. This avoids faster-whisper's own PyAV decoding path.
The model is downloaded from Hugging Face on first load (internet needed once), then cached.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from errors import (EmptyFileError, NoSpeechError, UnreadableAudioError,
                    UnsupportedFileError)

logger = logging.getLogger("uvicorn.error")

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small.en")  # tiny.en / base.en / small.en / medium.en
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")          # auto / cpu / cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 is fastest on CPU; use float16 on GPU
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}

_model = None
_model_lock = threading.Lock()       # guards model loading
_transcribe_lock = threading.Lock()  # one transcription at a time (shared model, high RAM use)


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel  # lazy import: slow to load
            _model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=os.cpu_count() or 4,
            )
    return _model


def validate_audio_file(path) -> Path:
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{p.suffix}'. Please upload one of: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    if not p.exists():
        raise UnreadableAudioError("The uploaded file could not be found on the server.")
    if p.stat().st_size == 0:
        raise EmptyFileError("The uploaded file is empty.")
    return p


def _decode_with_ffmpeg(path: Path) -> np.ndarray:
    """Decode any supported audio/video file to 16 kHz mono float32 samples using ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise UnreadableAudioError(
            "ffmpeg was not found on the server. Install it, then restart the backend from a new terminal."
        )
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
           "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg could not decode %s: %s", path.name, e.stderr.decode(errors="ignore")[-300:])
        raise UnreadableAudioError(
            "The audio file could not be read. It may be corrupted or in an unsupported encoding."
        ) from e
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def transcribe_audio(path) -> str:
    """Return the raw transcript text, or raise a PipelineError subclass.

    Only genuine decode failures are reported as an unreadable file. Model load problems
    and other unexpected errors propagate so they are logged with a traceback.
    """
    p = validate_audio_file(path)

    audio = _decode_with_ffmpeg(p)
    if audio.size == 0:
        raise NoSpeechError("No speech was detected in the recording.")

    model = _get_model()  # a load failure is not a bad-file error

    started = time.perf_counter()
    with _transcribe_lock:
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=1,                       # greedy decoding: much faster, near-identical quality
            vad_filter=True,                   # skip silence / non-speech
            condition_on_previous_text=False,  # reduces hallucinated text on quiet sections
        )
        # `segments` is a lazy generator: the real decoding happens while it is consumed.
        text = " ".join(s.text.strip() for s in segments).strip()

    logger.info("Transcribed %s (%.0fs of audio) in %.1fs", p.name, audio.size / 16000, time.perf_counter() - started)

    if not text:
        raise NoSpeechError("No speech was detected in the recording.")
    return text
