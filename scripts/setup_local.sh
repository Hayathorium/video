#! /bin/bash
# One-time setup for running RealVideo fully locally (no ZAI cloud API):
#   - installs llama.cpp's prebuilt `llama` CLI (LLM server)
#   - downloads Qwen2.5-7B-Instruct Q4_K_M GGUF
#   - creates .venv_tts with torch (CUDA 12.x) + `qwen-tts` (Qwen3-TTS)
#
# NOTE: TTS uses the `qwen-tts` package directly instead of vLLM-Omni. vLLM
# 0.28.x wheels only ship cu129/cu130, but this machine's driver (570.x) tops
# out at CUDA 12.8, so vLLM cannot run here.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="models"
VENV=".venv_tts"
GGUF_BASE="qwen2.5-7b-instruct-q4_k_m"
GGUF_REPO="Qwen/Qwen2.5-7B-Instruct-GGUF"

# 1. llama.cpp (prebuilt binary via the official installer, no CUDA build needed)
if [ ! -x "$HOME/.local/bin/llama" ]; then
    echo "==> Installing llama.cpp (prebuilt CUDA binary) ..."
    curl -LsSf https://llama.app/install.sh | sh
fi

# 2. GGUF model (Q4_K_M, split into 2 shards)
mkdir -p "$MODEL_DIR"
for shard in 00001-of-00002 00002-of-00002; do
    f="$MODEL_DIR/${GGUF_BASE}-${shard}.gguf"
    if [ ! -s "$f" ]; then
        echo "==> Downloading ${GGUF_BASE}-${shard}.gguf ..."
        hf download "$GGUF_REPO" "${GGUF_BASE}-${shard}.gguf" \
            --local-dir "$MODEL_DIR"
    fi
done

# 3. TTS venv (torch cu126 + qwen-tts)
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip wheel -q
# NOTE: cu126 wheels lack Blackwell (sm_120) kernels ("no kernel image available on
# the device" at load time on RTX PRO 6000 Blackwell). Use the default PyPI wheels
# instead, which ship a current cu13x build with Blackwell support.
"$VENV/bin/pip" install -U torch torchaudio
"$VENV/bin/pip" install "qwen-tts==0.1.1"
"$VENV/bin/pip" install hf_transfer

# 4. System deps for qwen-tts (libsox)
if ! command -v sox >/dev/null 2>&1; then
    echo "==> Installing libsox (apt) ..."
    apt-get update -y
    apt-get install -y sox libsox-dev libsox-fmt-all
fi

echo "==> Setup complete."
echo "    LLM server : llama serve"
echo "    TTS server : $VENV/bin/python scripts/tts_server.py"
echo "    Run        : bash scripts/run_local.sh"
