#!/usr/bin/env bash
# Serve GLM-5.3 (NVFP4, TP2) inside the GLM-5.3 MoET container.
#
# MOET_PROFILE selects the MoET cache tiering profile:
#   cache    - RAM planes-cache + NVMe moe-store with BASE_CACHE_GB/BASE_RAM_GB
#   resident - all expert planes resident on GPU; the base-cache/store env
#              controls are omitted entirely (never set to 0).
set -euo pipefail

IMAGE="${IMAGE:-glm53-moet:local}"
MODEL_PATH="${MODEL_PATH:-dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4}"
REVISION="835b767e640aeaace97bd9d8b6d4ddecd9d8e9d4"
MOET_PROFILE="${MOET_PROFILE:-cache}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PLANES_CACHE="${PLANES_CACHE:-$HOME/.cache/glm53/planes-cache}"
MOE_STORE="${MOE_STORE:-$HOME/.cache/glm53/moe-store}"

PORT="${PORT:-8000}"
HOST_BIND="${HOST_BIND:-127.0.0.1}"
SHM_SIZE="${SHM_SIZE:-64g}"

MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

case "${MOET_PROFILE}" in
    cache | resident) ;;
    *)
        echo "error: MOET_PROFILE must be 'cache' or 'resident' (got '${MOET_PROFILE}')" >&2
        exit 1
        ;;
esac

case "${ENFORCE_EAGER}" in
    0 | 1) ;;
    *)
        echo "error: ENFORCE_EAGER must be 0 or 1 (got '${ENFORCE_EAGER}')" >&2
        exit 1
        ;;
esac

cli=(
    "${MODEL_PATH}"
    --revision "${REVISION}"
    --tensor-parallel-size 2
    --quantization modelopt
    --moe-backend marlin
    --kv-cache-dtype fp8
    --max-model-len 32768
    --max-num-seqs 1
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --reasoning-parser glm45
    --tool-call-parser glm47
    --enable-auto-tool-choice
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    cli+=(--enforce-eager)
fi

cli+=("$@")

moe_env=()
if [[ "${MOET_PROFILE}" == "cache" ]]; then
    BASE_CACHE_GB="${BASE_CACHE_GB:-32}"
    BASE_RAM_GB="${BASE_RAM_GB:-32}"
    moe_env+=(
        -e "VLLM_MOE_W2_BASE_CACHE_GB=${BASE_CACHE_GB}"
        -e "VLLM_MOE_W2_BASE_RAM_GB=${BASE_RAM_GB}"
        -e VLLM_MOE_W2_STORE_DIR=/moe-store
    )
fi

docker run \
    --rm \
    --gpus all \
    --ipc=host \
    --shm-size "${SHM_SIZE}" \
    -p "${HOST_BIND}:${PORT}:8000" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${PLANES_CACHE}:/planes-cache" \
    -v "${MOE_STORE}:/moe-store" \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_MOE_W2=1 \
    -e VLLM_MOE_W2_CUBIT_DIR=/cubit-share \
    -e VLLM_MOE_W2_KS=4096,1024 \
    -e VLLM_MOE_W2_PLANES_CACHE=/planes-cache \
    -e VLLM_MOE_W2_CKPT_ID="${REVISION}" \
    -e VLLM_MOE_W2_DELTA=1 \
    -e VLLM_MOE_W2_DELTA_GB=2 \
    -e VLLM_MOE_W2_GATE=0 \
    -e VLLM_MOE_W2_BASE_MISS_TOL=0 \
    -e VLLM_MOE_W2_REPLAY_MODE=strict \
    "${moe_env[@]}" \
    "${IMAGE}" \
    "${cli[@]}"
