# RealVideo — Groq LLM + Local TTS

RealVideo is a WebSocket-based video calling system: text input in, an audio reply out, lip-synced
in real time via autoregressive diffusion.

This fork replaces the original cloud **ZAI API** (`GLM-4.5-AirX` + `GLM-TTS`) with:

- **LLM**: [Groq](https://console.groq.com/) (`openai/gpt-oss-20b`), OpenAI-compatible, cloud — keeps
  the LLM off the local GPU. Requires `GROQ_API_KEY`.
- **TTS**: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (0.6B CustomVoice) served locally via
  [`qwen-tts`](https://pypi.org/project/qwen-tts/) — no API key needed.

## Architecture

`scripts/run_local.sh` starts two processes:

| Service | Tool | Port | Model |
|---|---|---|---|
| LLM | Groq API (cloud) | — | `openai/gpt-oss-20b` |
| TTS | `qwen-tts` FastAPI wrapper | 8091 | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| App | `torchrun app.py` (DiT + VAE) | 8003 | `Wan2.2-S2V-14B` |

The app (rank 0: VAE/text/audio encoders; rank 1+: DiT sequence-parallel workers) needs at least 2
ranks. On a **single GPU**, both ranks share that one device (DiT sequence-parallel size 1). On
**2+ GPUs**, each rank gets its own GPU, sequence-parallel over the DiT; TTS runs on GPU 0.

## Requirements

- **1+ GPU, ≥80 GB** (Blackwell RTX PRO 6000, H100/H200, ...). 2+ GPUs with NVLink lower per-block
  latency via sequence parallelism (see [Reference timing](#reference-timing)).
- A [Groq API key](https://console.groq.com/keys) (`GROQ_API_KEY`).
- Python 3.10–3.12, `pip3`.
- CUDA driver 570.x/580.x+ (CUDA 12.8+).
- A modern browser (WebSocket + Web Audio API).
- `sox`/`libsox-dev` (installed by `setup_local.sh`).
- **flash-attn (required)** — without it, the DiT's cross-attention crashes silently on every
  generation request (LLM/TTS still respond, but the avatar never speaks). Install a prebuilt wheel
  from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases)
  matching your torch/CUDA/Python (avoids a 20–40 min source build):
  ```bash
  source /venv/main/bin/activate
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
  No matching wheel? Pin torch to a version the release matrix covers (e.g. the `torch==2.7.1`
  already in `requirements.txt`) rather than falling back to a source build; only build from source
  (`MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="<arch>" pip3 install flash-attn --no-build-isolation`) as a
  last resort.
- **Blackwell GPUs (sm_120):** the PyPI `torch==2.7.1`/`torchvision==0.22.1` wheels predate Blackwell
  support and fail at load time. Install the same pinned versions from the `cu128` index instead, in
  both the main env and `.venv_tts`:
  ```bash
  pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  ```

> **Note on vLLM**: this repo intentionally skips vLLM-Omni for TTS — its 0.28.x wheels need a CUDA
> ≥12.9 driver, which a 570.x (CUDA 12.8) driver can't satisfy. `qwen-tts` runs the same model without it.

## Quick Start

### 1. Download the lip-sync model

```bash
hf download Wan-AI/Wan2.2-S2V-14B --local-dir wan_models/Wan2.2-S2V-14B \
    --exclude "wav2vec2-large-xlsr-53-english/*" --exclude "diffusion_pytorch_model*.safetensors"
hf download zai-org/RealVideo model.pt --local-dir .
```

(Use `hf download`, not the deprecated `huggingface-cli`.) Both `--exclude` patterns are safe: the
wav2vec2 folder is fetched separately at runtime, and the DiT weights load from `model.pt` instead of
the safetensors shards (`config/config.py`'s `PATH_TO_YOUR_MODEL`) — together they save ~33 GB.

**Known path mismatch:** `self_forcing/utils/wan_wrapper.py` expects the VAE at
`wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`, which the download above doesn't create. Symlink it:

```bash
mkdir -p wan_models/Wan2.1-T2V-1.3B
ln -s "$(pwd)/wan_models/Wan2.2-S2V-14B/Wan2.1_VAE.pth" wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
pip3 install librosa aiohttp orjson torchaudio==2.7.1  # imported but missing from requirements.txt
```

### 3. One-time local TTS setup

```bash
bash scripts/setup_local.sh   # creates .venv_tts with qwen-tts; model downloads on first serve
```

### 4. Set your Groq API key

```bash
echo 'GROQ_API_KEY=gsk_...' >> .env   # or export it; .env is git-ignored
```

### 5. Start the service

```bash
bash scripts/run_local.sh
```

Starts the local TTS server (8091), waits for it to be healthy, then the app (8003). GPU count is
auto-detected to pick `torchrun --nproc_per_node`; prefix with `CUDA_VISIBLE_DEVICES=0,1` to select
specific GPUs.

### 6. Access the application

Open **http://localhost:8003**.

## Usage

1. **Set avatar**: upload an image (or keep the default `resources/cat.png`).
2. **Connect**: click "Connect" to open the WebSocket.
3. **Send a message**: type text and press Enter / click "Send".
4. **Watch the reply**: the lip-synced video plays on the left.

## Voices

Predefined Qwen3-TTS CustomVoice set (no voice cloning in this mode): `vivian`, `ryan`, `aiden`,
`dylan`, `eric`, `ono_anna`, `serena`, `sohee`, `uncle_fu`.

## Tuning resolution for speed

Video resolution is set by `max_image_area` in `read_image()` (`core/lip_sync.py`) — default `16384`
(≈128×128) — not by `config/config.py`'s `VideoConfig` fields (currently unused). Lower for speed,
raise for quality; `65536` (≈256×256) is a reasonable middle ground.

## Reference timing

Time (ms) for the DiT to generate one block; under **500 ms** is smooth real-time. Parenthesized
values are with `torch.compile` enabled.

| DiT sp size / Denoising steps | 2                         | 4                     |
|-------------------------------|---------------------------|-----------------------|
| 1                             | 563.84 ms (**442.61 ms**) | 943.13 ms (723.06 ms) |
| 2                             | **384.86 ms**             | 655.92 ms (527.11 ms) |
| 4                             | **306.39 ms**             | 513.72 ms (**480.68 ms**) |

## Notes

- Only **one** WebSocket client at a time; a second connection is rejected until the first closes.
- Send an `image_config` message before any `text`/`audio` message — the DiT won't start otherwise.
- `scripts/ws_test.py` is a headless smoke test for the full text → LLM → TTS → lip-sync pipeline.
- `GROQ_API_KEY` must be set before starting; `run_local.sh` exits immediately if it isn't.

## Acknowledgements

- [Groq](https://groq.com/) (`openai/gpt-oss-20b`)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / [`qwen-tts`](https://pypi.org/project/qwen-tts/)
- [self forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan2.2-S2V](https://github.com/Wan-Video/Wan2.2)
- [RealVideo](https://github.com/zai-org/RealVideo)
