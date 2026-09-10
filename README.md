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

- **2+ GPUs**, ≥80 GB each (e.g. H100 / H200, or Blackwell RTX PRO 6000), with NVLink.
- Python 3.10–3.12, `pip3`.
- CUDA driver (tested on 570.x/580.x, i.e. CUDA 12.8+).
- A modern browser (WebSocket + Web Audio API).
- `sox`/`libsox-dev`, `ninja-build`, `cmake` (`ninja-build` is needed for the
  `llama.cpp` CUDA build in step 3; `sox`/`libsox-dev` are installed
  automatically by `setup_local.sh`).
- **flash-attn is required, not optional.** The TTS server prints "flash-attn
  is not installed, will only run the manual PyTorch version" and still
  works without it, but the main DiT model
  (`self_forcing/wan/modules/model.py`) calls `flash_attention()` with no
  fallback path — without flash-attn installed, every real generation
  request crashes silently inside the DiT worker process (the LLM and TTS
  still respond, and the avatar stays on screen showing only the static
  reference frame, but it never actually answers). Install it after step 2:
  ```bash
  MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST="<your arch, e.g. 12.0 for Blackwell>" \
      pip3 install flash-attn --no-build-isolation
  ```
  Building from source can take a while; restricting `TORCH_CUDA_ARCH_LIST`
  to your actual GPU's compute capability (`python3 -c "import torch;
  print(torch.cuda.get_device_capability(0))"`) cuts the build time a lot.
- **On Blackwell GPUs (sm_120, e.g. RTX PRO 6000):** `requirements.txt`
  pins `torch==2.7.1`/`torchvision==0.22.1` with no CUDA-variant suffix,
  which resolves to a PyPI wheel that predates Blackwell support and fails
  at model-load time with `CUDA error: no kernel image is available for
  execution on the device`. Skip those two pins when installing
  `requirements.txt` and instead install a Blackwell-capable build, e.g.:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```
  Do this for **both** the main environment and `.venv_tts` (created by
  `setup_local.sh`) — the script already installs an unpinned
  `torch`/`torchaudio` into `.venv_tts` for this reason.

> **Note on vLLM**: this repo deliberately does **not** use vLLM-Omni for TTS. vLLM 0.28.x wheels
> only ship `cu129`/`cu130` builds, which require a CUDA ≥12.9 driver; on a 570.x (CUDA 12.8) driver
> they fail with "driver too old". `qwen-tts` runs the same model without vLLM.

## Quick Start

### 1. Download the lip-sync model

`huggingface-cli` is deprecated on recent `huggingface_hub` versions and
exits immediately with a "no longer works" message — use `hf download`
instead (installed by `pip3 install huggingface_hub` /
`requirements.txt`):

```bash
hf download Wan-AI/Wan2.2-S2V-14B --local-dir wan_models/Wan2.2-S2V-14B
hf download zai-org/RealVideo model.pt --local-dir .
```

`Wan2.2-S2V-14B` also includes a `wav2vec2-large-xlsr-53-english/` folder;
it's unused (`AudioEncoder` downloads its own copy from the HF hub at
runtime) and safe to skip with `--exclude "wav2vec2-large-xlsr-53-english/*"`
to save ~5 GB.

Then set the checkpoint path in `config/config.py`:

```python
PATH_TO_YOUR_MODEL = "model.pt"  # or the absolute path to your checkpoint
```

**Known path mismatch:** `self_forcing/utils/wan_wrapper.py` hardcodes the
VAE checkpoint path as `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`, but the
download above only creates `wan_models/Wan2.2-S2V-14B/` (which already
contains its own copy of `Wan2.1_VAE.pth`). Rather than downloading the
whole separate `Wan2.1-T2V-1.3B` repo, symlink it in:

```bash
mkdir -p wan_models/Wan2.1-T2V-1.3B
ln -s "$(pwd)/wan_models/Wan2.2-S2V-14B/Wan2.1_VAE.pth" wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

A few gaps to watch for, depending on your Python/torch versions:

- `numpy==1.24.4` (pinned in `requirements.txt`) doesn't support Python
  3.12 (it needs `numpy.distutils`, removed in 3.12) and fails to build.
  Use `numpy>=1.26` instead on Python 3.12.
- If you bump `numpy` to 2.x (e.g. because something else pulls in
  `librosa>=1.0`, which requires numpy 2), also upgrade `opencv-python`
  past `4.8.0.74` — that version was built against the numpy 1.x ABI and
  fails with `numpy.core.multiarray failed to import` otherwise.
- `librosa`, `aiohttp`, and `orjson` are imported by the app but missing
  from `requirements.txt`; install them too:
  ```bash
  pip3 install librosa aiohttp orjson
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

## Tuning resolution for speed

The streamed video's resolution isn't controlled by `config/config.py`'s
`VideoConfig.frame_width`/`frame_height` (those fields are currently
unused). It's set by `max_image_area` in `read_image()`
(`core/lip_sync.py`), which resizes the uploaded reference image to fit
that many total pixels (default `16384` ≈ 128×128, snapped to the nearest
16:9/4:3/1:1 bucket and multiple of 64) regardless of the image's original
size. Lowering it reduces DiT and VAE compute per block; raising it
improves quality. `65536` (≈256×256) is a reasonable quality/speed middle
ground if the default looks too degraded.

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
