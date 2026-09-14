#! /bin/bash
# One-time setup for running RealVideo with the LLM on Groq's cloud API and
# TTS fully local:
#   - creates .venv_tts with torch (CUDA 12.x) + `qwen-tts` (Qwen3-TTS)
#
# NOTE: TTS uses the `qwen-tts` package directly instead of vLLM-Omni. vLLM
# 0.28.x wheels only ship cu129/cu130, but this machine's driver (570.x) tops
# out at CUDA 12.8, so vLLM cannot run here.
#
# The LLM no longer runs locally: it calls Groq's OpenAI-compatible API
# (`openai/gpt-oss-20b`), so no GGUF download/llama.cpp install is needed and
# no GPU memory is spent on it. Set GROQ_API_KEY before running the app.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv_tts"

# 1. TTS venv (torch + qwen-tts)
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

# 2. System deps for qwen-tts (libsox)
if ! command -v sox >/dev/null 2>&1; then
    echo "==> Installing libsox (apt) ..."
    apt-get update -y
    apt-get install -y sox libsox-dev libsox-fmt-all
fi

echo "==> Setup complete."
echo "    LLM server : Groq API (set GROQ_API_KEY; no local install needed)"
echo "    TTS server : $VENV/bin/python scripts/tts_server.py"
echo "    Run        : bash scripts/run_local.sh"
