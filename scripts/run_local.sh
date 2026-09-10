#! /bin/bash
# Launch RealVideo fully locally: llama.cpp LLM (8080) + Qwen3-TTS (8091) + app.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

function get_gpu_count() {
    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --list-gpus | wc -l
        else
            echo "0"
        fi
    elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "0"
    else
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    fi
}

export GLOO_SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-}
export NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-}
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=./.inductor_cache
export TORCHDYNAMO_VERBOSE=1
export GPU_COUNT=$(get_gpu_count)

# Single-node 2x H100 NVL: disable InfiniBand/PXN and use NVLink P2P only.
export NCCL_IB_DISABLE=1
export NCCL_PXN_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_LEVEL=NVL
# This environment's CUDA VMM API (cuMemCreate) returns INVALID_VALUE; disable NCCL's cuMem allocator.
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG=VERSION

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

LLM_PORT=8080
TTS_PORT=8091
LLM_MODEL="models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# --- 1. llama.cpp LLM server (GPU 0) ---
echo "Starting llama-server on :$LLM_PORT ..."
CUDA_VISIBLE_DEVICES=0 llama.cpp/build/bin/llama-server \
    -m "$LLM_MODEL" \
    --host 127.0.0.1 --port "$LLM_PORT" \
    -ngl 99 -c 4096 \
    > logs/llama.log 2>&1 &
LLAMA_PID=$!

# --- 2. Qwen3-TTS server (GPU 0) ---
echo "Starting Qwen3-TTS server on :$TTS_PORT ..."
CUDA_VISIBLE_DEVICES=0 .venv_tts/bin/python scripts/tts_server.py \
    --host 127.0.0.1 --port "$TTS_PORT" \
    > logs/tts.log 2>&1 &
TTS_PID=$!

cleanup() {
    echo "Shutting down local services ..."
    kill "$LLAMA_PID" "$TTS_PID" 2>/dev/null || true
    wait "$LLAMA_PID" "$TTS_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for LLM ---
echo -n "Waiting for llama-server"
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1; then
        echo " ready."; break
    fi
    echo -n "."; sleep 1
done

# --- wait for TTS (first run downloads the model; be patient) ---
echo -n "Waiting for Qwen3-TTS"
for _ in $(seq 1 1800); do
    if curl -sf "http://127.0.0.1:$TTS_PORT/health" >/dev/null 2>&1; then
        echo " ready."; break
    fi
    echo -n "."; sleep 1
done

echo "Starting RealVideo app on $GPU_COUNT GPUs ..."
torchrun --standalone --nproc_per_node="$GPU_COUNT" app.py
