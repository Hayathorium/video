#! /bin/bash
# Launch RealVideo: Groq cloud LLM + local Qwen3-TTS (8091) + app.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# Pick up GROQ_API_KEY (and any other overrides) from a .env file in the repo
# root, if present, without clobbering already-exported values.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "GROQ_API_KEY is not set. Get a key at https://console.groq.com/keys and export it:" >&2
    echo "    export GROQ_API_KEY=gsk_..." >&2
    exit 1
fi

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
# Without this, current_start.item() in causal_model_s2v.py forces a graph
# break on every block (torch.compile's own warning suggests this fix),
# splitting the compiled model into separate fused regions with dispatch
# overhead between them.
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
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

TTS_PORT=8091

# The service needs at least 2 ranks (rank 0: VAE/interface, rank 1+: DiT
# worker(s), sequence-parallel over the DiT ranks). On a single GPU, run the
# minimum of 2 ranks sharing that one device (a degenerate sequence-parallel
# group of size 1 for the DiT worker) instead of the process-per-GPU mapping.
if [ "$GPU_COUNT" -le 1 ]; then
    NPROC_PER_NODE=2
    # Two ranks on the same physical device: force the NCCL IPC/SHM path
    # instead of the (inter-device) P2P path.
    export NCCL_P2P_DISABLE=1
    echo "Only $GPU_COUNT GPU visible: running the app rank and the DiT worker" \
        "on the same device (world size 2, DiT sequence-parallel size 1)."
else
    NPROC_PER_NODE=$GPU_COUNT
fi

# --- Qwen3-TTS server (local, GPU 0) ---
echo "Starting Qwen3-TTS server on :$TTS_PORT ..."
CUDA_VISIBLE_DEVICES=0 .venv_tts/bin/python scripts/tts_server.py \
    --host 127.0.0.1 --port "$TTS_PORT" \
    > logs/tts.log 2>&1 &
TTS_PID=$!

cleanup() {
    echo "Shutting down local services ..."
    kill "$TTS_PID" 2>/dev/null || true
    wait "$TTS_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for TTS (first run downloads the model; be patient) ---
echo -n "Waiting for Qwen3-TTS"
for _ in $(seq 1 1800); do
    if curl -sf "http://127.0.0.1:$TTS_PORT/health" >/dev/null 2>&1; then
        echo " ready."; break
    fi
    echo -n "."; sleep 1
done

echo "Starting RealVideo app ($NPROC_PER_NODE ranks, $GPU_COUNT GPU(s)) ..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" app.py
