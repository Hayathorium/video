#! /bin/bash
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

export GLOO_SOCKET_IFNAME=$MLP_SOCKET_IFNAME
export NCCL_SOCKET_IFNAME=$MLP_SOCKET_IFNAME
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

echo "Starting Standalone Torchrun on $GPU_COUNT GPUs..."

torchrun \
    --standalone \
    --nproc_per_node=$GPU_COUNT \
    app.py