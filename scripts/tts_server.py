"""Local Qwen3-TTS server (OpenAI /v1/audio/speech compatible).

Serves the Qwen3-TTS-12Hz-0.6B-CustomVoice model directly via the `qwen-tts`
package (no vLLM), returning raw 16-bit signed little-endian mono PCM at
24 kHz. This is the local replacement for the ZAI `glm-tts` cloud API.

Run with the app venv's torch (CUDA 12.x, driver 570) and `qwen-tts` installed:
    CUDA_VISIBLE_DEVICES=0 .venv_tts/bin/python scripts/tts_server.py --port 8091
"""

import argparse
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("tts_server")

MODEL_ID = os.getenv("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

# Qwen3-TTS CustomVoice speaker names, keyed by the lowercase ids the
# RealVideo frontend/backend use.
SPEAKERS = {
    "vivian": "Vivian",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "dylan": "Dylan",
    "eric": "Eric",
    "ono_anna": "Ono_Anna",
    "serena": "Serena",
    "sohee": "Sohee",
    "uncle_fu": "Uncle_Fu",
}

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    import torch

    from qwen_tts import Qwen3TTSModel

    logger.info("Loading Qwen3-TTS model %s ...", MODEL_ID)
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    logger.info("Qwen3-TTS model loaded.")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Qwen3-TTS local server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/audio/voices")
async def list_voices():
    return {"voices": list(SPEAKERS.keys())}


@app.post("/v1/audio/speech")
async def speech(request: Request):
    body = await request.json()
    text = body.get("input") or body.get("text") or ""
    voice = str(body.get("voice", "vivian")).lower()
    language = body.get("language", "Auto")
    response_format = body.get("response_format", "pcm")
    if not text:
        return JSONResponse({"error": "empty input"}, status_code=400)

    speaker = SPEAKERS.get(voice, "Vivian")
    # `qwen-tts` accepts "Auto" for language auto-detection.
    if not language:
        language = "Auto"

    model = load_model()
    wavs, sr = model.generate_custom_voice(
        text=text, language=language, speaker=speaker
    )

    if isinstance(wavs, (list, tuple)):
        wav = wavs[0]
    else:
        wav = wavs
    wav = np.asarray(wav, dtype=np.float32)
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")

    if response_format == "pcm":
        return Response(content=pcm.tobytes(), media_type="application/octet-stream")

    # Fallback: wrap PCM in a minimal WAV container (16-bit mono).
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
