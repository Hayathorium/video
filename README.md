# RealVideo — Local LLM + TTS

RealVideo is a WebSocket-based video calling system that takes text input, generates an audio
response, and uses autoregressive diffusion to produce a real-time lip-synced video. The system is
modular with a clean code structure.

This fork replaces the cloud **ZAI API** (`GLM-4.5-AirX` + `GLM-TTS`) with **fully local** inference,
so no API key is required:

- **LLM**: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` running
  `Qwen2.5-7B-Instruct` (GGUF).
- **TTS**: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (0.6B CustomVoice) served by the
  [`qwen-tts`](https://pypi.org/project/qwen-tts/) package.

## Architecture

Three processes run together (orchestrated by `scripts/run_local.sh`):

| Service | Tool | Port | Model |
|---|---|---|---|
| LLM | `llama-server` (OpenAI-compatible) | 8080 | `Qwen2.5-7B-Instruct` Q4_K_M GGUF |
| TTS | `qwen-tts` FastAPI wrapper | 8091 | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| App | `torchrun app.py` (DiT + VAE) | 8003 | `Wan2.2-S2V-14B` |

The app itself still uses two GPUs: one for the VAE service, the rest for parallel DiT inference.
The LLM and TTS servers are pinned to GPU 0.

## Requirements

- **2+ GPUs**, ≥80 GB each (e.g. H100 / H200), with NVLink.
- Python 3.10–3.12, `pip3`.
- CUDA driver (tested on 570.x, i.e. CUDA 12.8).
- A modern browser (WebSocket + Web Audio API).
- `sox`/`libsox-dev` (installed automatically by `setup_local.sh`).

> **Note on vLLM**: this repo deliberately does **not** use vLLM-Omni for TTS. vLLM 0.28.x wheels
> only ship `cu129`/`cu130` builds, which require a CUDA ≥12.9 driver; on a 570.x (CUDA 12.8) driver
> they fail with "driver too old". `qwen-tts` runs the same model without vLLM.

## Quick Start

### 1. Download the lip-sync model

```bash
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir-use-symlinks False --local-dir wan_models/Wan2.2-S2V-14B
huggingface-cli download zai-org/RealVideo model.pt --local-dir .
```

Then set the checkpoint path in `config/config.py`:

```python
PATH_TO_YOUR_MODEL = "model.pt"  # or the absolute path to your checkpoint
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 3. One-time local LLM/TTS setup

Builds llama.cpp, downloads the Qwen2.5-7B GGUF, and creates `.venv_tts` with `qwen-tts`:

```bash
bash scripts/setup_local.sh
```

The `Qwen3-TTS-12Hz-0.6B-CustomVoice` model is downloaded automatically on first TTS serve.

### 4. Start the service

```bash
bash scripts/run_local.sh
```

This launches the LLM server (port 8080), the TTS server (port 8091), waits for both to become
healthy, then starts the app on port 8003. To select specific GPUs, prefix with
`CUDA_VISIBLE_DEVICES=0,1`.

### 5. Access the application

Open **http://localhost:8003**.

## Usage

1. **Set avatar**: upload an image (or keep the default `resources/cat.png`).
2. **Connect**: click "Connect" to open the WebSocket.
3. **Send a message**: type text and press Enter / click "Send".
4. **Watch the reply**: the lip-synced video response plays on the left.

## Voices

The local TTS uses the predefined Qwen3-TTS CustomVoice set (voice cloning is not supported in this
mode): `vivian`, `ryan`, `aiden`, `dylan`, `eric`, `ono_anna`, `serena`, `sohee`, `uncle_fu`.

## Reference timing

The table below shows reference times (in ms) for the DiT to generate one block. Within **500 ms**,
smooth real-time generation is achievable. Numbers in parentheses indicate the time with compilation
enabled.

| DiT sp size / Denoising steps | 2                         | 4                     |
|-------------------------------|---------------------------|-----------------------|
| 1                             | 563.84 ms (**442.61 ms**) | 943.13 ms (723.06 ms) |
| 2                             | **384.86 ms**             | 655.92 ms (527.11 ms) |
| 4                             | **306.39 ms**             | 513.72 ms (**480.68 ms**) |

## Notes

- Only **one** WebSocket client can connect at a time; a second connection is rejected until the
  first closes.
- The WebSocket handshake requires an `image_config` message (the reference image) before any
  `text`/`audio` message, otherwise the DiT never starts generating.
- `scripts/ws_test.py` is a headless smoke-test client that exercises the full
  text → LLM → TTS → lip-sync pipeline.

## Acknowledgements

This project utilizes the following open-source libraries and models:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / [`qwen-tts`](https://pypi.org/project/qwen-tts/)
- [self forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan2.2-S2V](https://github.com/Wan-Video/Wan2.2)
- [RealVideo](https://github.com/zai-org/RealVideo)
