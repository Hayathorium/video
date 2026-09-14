# RealVideo — Groq LLM + Local TTS

RealVideo is a WebSocket-based video calling system that takes text input, generates an audio
response, and uses autoregressive diffusion to produce a real-time lip-synced video. The system is
modular with a clean code structure.

This fork replaces the original cloud **ZAI API** (`GLM-4.5-AirX` + `GLM-TTS`) with:

- **LLM**: [Groq](https://console.groq.com/)'s OpenAI-compatible cloud API, running
  `openai/gpt-oss-20b`. Offloading the LLM to Groq keeps it off the local GPU entirely, which is
  what makes the single-GPU setup below possible. Requires a `GROQ_API_KEY`.
- **TTS**: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (0.6B CustomVoice) served locally by the
  [`qwen-tts`](https://pypi.org/project/qwen-tts/) package (no API key required).

## Architecture

Two processes run together (orchestrated by `scripts/run_local.sh`):

| Service | Tool | Port | Model |
|---|---|---|---|
| LLM | Groq API (cloud, OpenAI-compatible) | — | `openai/gpt-oss-20b` |
| TTS | `qwen-tts` FastAPI wrapper | 8091 | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| App | `torchrun app.py` (DiT + VAE) | 8003 | `Wan2.2-S2V-14B` |

The app process (rank 0: VAE/text/audio encoders, rank 1+: DiT sequence-parallel workers) runs on
**one or more GPUs**:

- **1 GPU**: `run_local.sh` starts `app.py` with 2 ranks (world size 2) both pinned to the same
  device — a degenerate sequence-parallel group of size 1 for the DiT worker. This is the
  minimum world size the architecture needs (rank 0 always does VAE/interface work, rank 1+ always
  does DiT), so a single GPU runs both roles as separate processes sharing the one device rather
  than one process per GPU. The TTS server also runs on that same GPU.
- **2+ GPUs**: one rank per GPU — rank 0 for VAE, the rest sequence-parallel over the DiT. The TTS
  server is pinned to GPU 0 alongside the VAE rank.

## Requirements

- **1+ GPU**, ≥80 GB (e.g. a Blackwell RTX PRO 6000, or an H100/H200). With 2+ GPUs and NVLink, the
  DiT can be split across them with sequence parallelism for lower per-block latency (see
  [Reference timing](#reference-timing)); a single GPU works but runs the DiT un-parallelized.
- A [Groq API key](https://console.groq.com/keys) (`GROQ_API_KEY`) for the LLM.
- Python 3.10–3.12, `pip3`.
- CUDA driver (tested on 570.x/580.x, i.e. CUDA 12.8+).
- A modern browser (WebSocket + Web Audio API).
- `sox`/`libsox-dev` (installed automatically by `setup_local.sh`) for the local TTS server.
- **flash-attn is required, not optional.** The TTS server prints "flash-attn
  is not installed, will only run the manual PyTorch version" and still
  works without it, but the main DiT model's cross-attention
  (`WanT2VCrossAttention.forward` in `self_forcing/wan/modules/model.py`,
  used unconditionally by the causal S2V blocks in
  `causal_model_s2v.py`) calls `flash_attention()` with no fallback path —
  without flash-attn installed, every real generation request crashes
  silently inside the DiT worker process (the LLM and TTS still respond,
  and the avatar stays on screen showing only the static reference frame,
  but it never actually answers).

  **Prefer a prebuilt wheel over building from source** — `pip3 install
  flash-attn` builds from source because PyPI only hosts the sdist, but the
  project's own GitHub releases page ships prebuilt wheels for common
  torch/CUDA/Python combos. Match one to your installed torch and skip the
  ~20-40 min compile:
  ```bash
  source /venv/main/bin/activate   # or your project venv
  TORCH_TAG=$(python3 -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))")
  ABI=$(python3 -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  PYTAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")

  curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
want = f'cu12torch${TORCH_TAG}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl'
for a in d['assets']:
    if want in a['name']:
        print(a['name']); print(a['browser_download_url']); break
" > /tmp/fa_asset.txt

  NAME=$(sed -n 1p /tmp/fa_asset.txt); URL=$(sed -n 2p /tmp/fa_asset.txt)
  [ -n "$NAME" ] && curl -L -o "$NAME" "$URL" && pip3 install "$NAME"
  ```
  If nothing matches (the release matrix currently tops out around torch
  2.9, so a bleeding-edge/unpinned torch install will usually miss it), pin
  torch to a version the matrix covers instead — e.g. the `torch==2.7.1`
  already pinned in `requirements.txt` — rather than falling back to a
  source build; see the Blackwell note below for how to get that pinned
  version with Blackwell support. Only build from source
  (`MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="<your arch, e.g. 12.0 for Blackwell>"
  pip3 install flash-attn --no-build-isolation`) if no matching wheel
  exists for your torch/CUDA/Python combo at all.
- **On Blackwell GPUs (sm_120, e.g. RTX PRO 6000):** `requirements.txt`
  pins `torch==2.7.1`/`torchvision==0.22.1` with no CUDA-variant suffix,
  which resolves to a PyPI wheel that predates Blackwell support and fails
  at model-load time with `CUDA error: no kernel image is available for
  execution on the device`. Skip those two pins when installing
  `requirements.txt` and instead install a Blackwell-capable build from the
  `cu128` index — **keep the same pinned versions** (rather than installing
  unpinned/latest) so a prebuilt flash-attn wheel still matches, per above:
  ```bash
  pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
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
hf download Wan-AI/Wan2.2-S2V-14B --local-dir wan_models/Wan2.2-S2V-14B \
    --exclude "wav2vec2-large-xlsr-53-english/*" --exclude "diffusion_pytorch_model*.safetensors"
hf download zai-org/RealVideo model.pt --local-dir .
```

`Wan2.2-S2V-14B` also includes a `wav2vec2-large-xlsr-53-english/` folder;
it's unused (`AudioEncoder` downloads its own copy from the HF hub at
runtime) and safe to skip, saving ~5 GB.

The `diffusion_pytorch_model-0000*-of-00004.safetensors` shards (the DiT
weights, ~28 GB) are also skippable: `core/dit_service.py` builds the DiT
(`WanDiffusionWrapper`) with `skip_init_model=True`, which only reads
`config.json` from `wan_models/Wan2.2-S2V-14B/` for the architecture — the
actual weights are loaded from `model.pt` instead
(`pipeline.generator.load_state_dict(state_dict["generator"])`). The
`config.json`, `models_t5_umt5-xxl-enc-bf16.pth` (T5 text encoder) and
`google/umt5-xxl/` (tokenizer) files are still required and are not excluded
by the pattern above.

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

### 3. One-time local TTS setup

Creates `.venv_tts` with `qwen-tts`:

```bash
bash scripts/setup_local.sh
```

The `Qwen3-TTS-12Hz-0.6B-CustomVoice` model is downloaded automatically on first TTS serve.

### 4. Set your Groq API key

Get a key from [console.groq.com/keys](https://console.groq.com/keys), then either export it or
drop it in a `.env` file in the repo root (both are picked up by `run_local.sh`; `.env` is
git-ignored):

```bash
echo 'GROQ_API_KEY=gsk_...' >> .env
```

### 5. Start the service

```bash
bash scripts/run_local.sh
```

This starts the local TTS server (port 8091), waits for it to become healthy, then starts the app
on port 8003. The LLM runs remotely on Groq, so there's nothing to launch or wait on for it.

On a single GPU, `app.py` runs with 2 ranks sharing that one device (see
[Architecture](#architecture)); this is automatic — `run_local.sh` detects the GPU count and picks
the right `torchrun --nproc_per_node`. On 2+ GPUs, prefix with `CUDA_VISIBLE_DEVICES=0,1` to select
specific ones.

### 6. Access the application

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
- `GROQ_API_KEY` must be set (env var or `.env`) before starting the service; `run_local.sh` exits
  immediately with a reminder if it isn't.

## Acknowledgements

This project utilizes the following open-source libraries and models:

- [Groq](https://groq.com/) (`openai/gpt-oss-20b`)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / [`qwen-tts`](https://pypi.org/project/qwen-tts/)
- [self forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan2.2-S2V](https://github.com/Wan-Video/Wan2.2)
- [RealVideo](https://github.com/zai-org/RealVideo)
