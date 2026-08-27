# MOET Provenance

This workspace is a source fork of vllm-project/vllm at the exact GLM-5.3
implementation commit, on the branch `glm53-moet`. The branch ports a
selected subset of the MoET runtime from the pinned references below onto
vLLM commit `933876c`: MoE W2 NVFP4 plane conversion, the base/delta
stores, strict V2 TP2 replay, and the SM120 cubins under
`kernels/cubins-sm120/`.

Excluded or not yet validated in this branch: expert parallelism and EPLB,
MTP, multimodal (vision), confidence gating, CUDA graph execution,
PP>1 base caching, and any end-to-end GPU or model accuracy run. See
`docs/models/glm53_moet.md` for the full phase-one boundary and the
validation order.

## Upstream base

- vLLM base commit (GLM-5.3 implementation):
  `933876c388fb129ad82590660e6506614559cb86`
- Upstream remote: `origin` -> https://github.com/vllm-project/vllm.git

## MoET reference remotes (read-only, fetchable)

- Kacper MoET source (patch source): https://github.com/kacper-daftcode/vLLM-Moet.git
  - Patch source commit (per `patch/SOURCE.txt`):
    `432f9d9cf4bd55cf458e7bc6efcdbcedf763e5f7`
  - Repository HEAD: `de0eb87e2d70e03dffcf6737cc51f36cf4323e16`
- jpezzulli runtime: https://github.com/jpezzulli/vllm.git
  - Runtime tip (branch `moet-v0.24.0`):
    `9b8460737e77c2f826dc5b5d9d918fe91a240feb`

## Pinned dependency / artifact identifiers

- Docker base digest:
  `sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4`
- FlashMLA commit:
  `8447acbcb558db892bf7c1197d225be1c95b168c`
- FlashAttention commit:
  `2b84100f50e1d2a8726a86c86d14c2f9c9e5a67c`
- Target checkpoint:
  `dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4` revision
  `835b767e640aeaace97bd9d8b6d4ddecd9d8e9d4`
