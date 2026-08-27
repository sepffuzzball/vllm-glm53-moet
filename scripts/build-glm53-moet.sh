#!/usr/bin/env bash
# Build the GLM-5.3 MoET container from the current source tree: first the
# vllm-openai base (docker/Dockerfile, target vllm-openai), then the thin
# MoET overlay (Dockerfile.glm53-moet) that layers in the SM120 cubins.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_IMAGE="${BASE_IMAGE:-glm53-vllm-openai:base}"
IMAGE="${IMAGE:-glm53-moet:local}"
CUDA_VERSION="${CUDA_VERSION:-13.0.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.17}"
PROGRESS="${PROGRESS:-auto}"

docker build \
    --progress "${PROGRESS}" \
    --target vllm-openai \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg FLASHINFER_VERSION="${FLASHINFER_VERSION}" \
    -t "${BASE_IMAGE}" \
    -f docker/Dockerfile \
    .

docker build \
    --progress "${PROGRESS}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE}" \
    -f Dockerfile.glm53-moet \
    .
