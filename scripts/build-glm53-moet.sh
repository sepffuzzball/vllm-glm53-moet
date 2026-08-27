#!/usr/bin/env bash
# Build the GLM-5.3 MoET container by overlaying this source tree's Python
# files onto the pinned vLLM image. The base image already contains the
# compiled native extensions, so this is a fast overlay build that reuses
# them instead of rebuilding vLLM from source (which used to take hours).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:glm53-flash@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4}"
IMAGE="${IMAGE:-glm53-moet:local}"
PROGRESS="${PROGRESS:-auto}"

echo "Building ${IMAGE} by overlaying Python files onto the pinned base image ${BASE_IMAGE}."
echo "The pinned base provides the compiled vLLM extensions, which this build reuses as-is; nothing is compiled here."

docker build \
    --progress "${PROGRESS}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE}" \
    -f Dockerfile.glm53-moet \
    .
