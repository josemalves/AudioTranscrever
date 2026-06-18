import gc
import os
import tempfile
from threading import Lock
from typing import Optional

import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "medium")
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "pt")
HF_TOKEN = os.environ.get("HF_TOKEN")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))

ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}

_models = {}
_align_model = None
_align_metadata = None
_align_language = None
_diarize = None
_lock = Lock()  # whisperx pipelines aren't thread-safe; serialize requests

app = FastAPI(title="WhisperX Service")


def get_model(name: str):
    if name not in _models:
        print(f"[whisperx] loading transcription model: {name}")
        _models[name] = whisperx.load_model(
            name,
            DEVICE,
            compute_type=COMPUTE_TYPE,
            language=DEFAULT_LANGUAGE,
        )
    return _models[name]


def get_align(language: str):
    global _align_model, _align_metadata, _align_language
    if _align_model is None or _align_language != language:
        print(f"[whisperx] loading alignment model for: {language}")
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code=language, device=DEVICE
        )
        _align_language = language
    return _align_model, _align_metadata


def get_diarize():
    global _diarize
    if _diarize is None:
        if not HF_TOKEN:
            raise HTTPException(
                status_code=400,
                detail="Diarização requer HF_TOKEN configurado no servidor.",
            )
        print("[whisperx] loading diarization pipeline")
        _diarize = whisperx.diarize.DiarizationPipeline(
            use_auth_token=HF_TOKEN, device=DEVICE
        )
    return _diarize


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_models.keys()),
        "device": DEVICE,
    }


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    diarize: str = Form("false"),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    diarize_flag = diarize.lower() in ("true", "1", "yes", "on")
    lang = (language or DEFAULT_LANGUAGE).strip()
    model_name = (model or DEFAULT_MODEL).strip()
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo não suportado: {model_name}. Escolhe entre: {sorted(ALLOWED_MODELS)}",
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        audio_path = tmp.name

    try:
        with _lock:
            audio = whisperx.load_audio(audio_path)

            tx_model = get_model(model_name)
            tx = tx_model.transcribe(audio, batch_size=BATCH_SIZE, language=lang)

            try:
                align_model, align_metadata = get_align(lang)
                aligned = whisperx.align(
                    tx["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False,
                )
            except Exception as e:
                print(f"[whisperx] alignment failed, using unaligned segments: {e}")
                aligned = {"segments": tx["segments"]}

            if diarize_flag:
                diarize_pipeline = get_diarize()
                diarize_segments = diarize_pipeline(audio_path)
                aligned = whisperx.assign_word_speakers(diarize_segments, aligned)

            duration = float(audio.shape[0]) / 16000.0

            segments = []
            text_parts = []
            for seg in aligned.get("segments", []):
                seg_text = (seg.get("text") or "").strip()
                segments.append(
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg_text,
                        "speaker": seg.get("speaker"),
                    }
                )
                if seg_text:
                    text_parts.append(seg_text)

            gc.collect()

            return {
                "text": " ".join(text_parts).strip(),
                "segments": segments,
                "duration": duration,
                "diarized": diarize_flag,
                "language": lang,
                "model": model_name,
            }
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
