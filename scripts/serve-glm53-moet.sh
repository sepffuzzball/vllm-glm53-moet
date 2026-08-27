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
REVISION="${REVISION:-835b767e640aeaace97bd9d8b6d4ddecd9d8e9d4}"
MODEL_IDENTITY="${MODEL_IDENTITY:-}"
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

# Classify MODEL_PATH. An existing host directory is bind-mounted read-only
# and passed as its in-container path (with no --revision); a repo ID is
# passed unchanged with --revision.
model_mounts=()
model_argv=()
ckpt_id=""

# Expand a leading ~ so the existence check works for tilde paths.
model_path_host="${MODEL_PATH}"
if [[ "${model_path_host}" == "~" ]]; then
    model_path_host="${HOME}"
elif [[ "${model_path_host}" == "~/"* ]]; then
    model_path_host="${HOME}/${model_path_host:2}"
fi

if [[ -d "${model_path_host}" ]]; then
    # Existing host directory (including a bare relative path like models/glm).
    model_real="$(realpath -e -- "${model_path_host}")"

    hf_cache_real=""
    if hf_cache_real_tmp="$(realpath -e -- "${HF_CACHE}" 2>/dev/null)"; then
        hf_cache_real="${hf_cache_real_tmp}"
    fi

    if [[ -n "${hf_cache_real}" ]] && [[ "${model_real}" == "${hf_cache_real}"/* ]]; then
        # HF cache snapshot: mount the whole cache so snapshot-to-blob
        # symlinks remain valid inside the container.
        model_mounts+=(-v "${HF_CACHE}:/model-hf-cache:ro")
        model_rel="${model_real#"${hf_cache_real}"/}"
        model_argv+=("/model-hf-cache/${model_rel}")

        snapshot_commit="${model_real##*/}"
        if [[ -z "${MODEL_IDENTITY}" ]] && [[ "${snapshot_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
            ckpt_id="${snapshot_commit}"
        else
            ckpt_id="${MODEL_IDENTITY}"
        fi
    else
        model_mounts+=(-v "${model_real}:/model:ro")
        model_argv+=("/model")
        ckpt_id="${MODEL_IDENTITY}"
    fi

    if [[ -z "${ckpt_id}" ]]; then
        echo "warning: MODEL_IDENTITY is unset; set it to an immutable artifact ID for safe persistent pack reuse" >&2
    fi
elif [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* || "${MODEL_PATH}" == ../* || "${MODEL_PATH}" == ~* ]]; then
    echo "error: MODEL_PATH does not exist on the host: '${MODEL_PATH}'" >&2
    exit 1
else
    # Remote Hugging Face repo ID. ckpt_id is intentionally left empty: the
    # runtime resolves the pinned REVISION to HF's commit hash as identity.
    if [[ "${MODEL_PATH}" =~ ^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$ ]]; then
        model_argv+=("${MODEL_PATH}" --revision "${REVISION}")
    else
        echo "error: MODEL_PATH is not a valid Hugging Face repo ID or host path: '${MODEL_PATH}'" >&2
        exit 1
    fi
fi

cli=(
    "${model_argv[@]}"
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

ckpt_env=()
if [[ -n "${ckpt_id}" ]]; then
    ckpt_env+=(-e "VLLM_MOE_W2_CKPT_ID=${ckpt_id}")
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
    -e VLLM_MOE_W2_DELTA=1 \
    -e VLLM_MOE_W2_DELTA_GB=2 \
    -e VLLM_MOE_W2_GATE=0 \
    -e VLLM_MOE_W2_BASE_MISS_TOL=0 \
    -e VLLM_MOE_W2_REPLAY_MODE=strict \
    "${model_mounts[@]}" \
    "${ckpt_env[@]}" \
    "${moe_env[@]}" \
    "${IMAGE}" \
    "${cli[@]}"
