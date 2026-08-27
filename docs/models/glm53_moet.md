# GLM-5.3 MoET on two RTX PRO 6000 GPUs

This fork is experimental. It has not been run against the full checkpoint or
on the target GPUs in this workspace. Do not treat it as production-ready or as
an accuracy or throughput claim.

## Pinned inputs

- vLLM source: `933876c388fb129ad82590660e6506614559cb86`
- Model: `dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4`
- Model revision: `835b767e640aeaace97bd9d8b6d4ddecd9d8e9d4`
- MoET lineage and kernel revisions: see `MOET_PROVENANCE.md`

Build and run:

```bash
scripts/build-glm53-moet.sh
scripts/serve-glm53-moet.sh
```

`scripts/build-glm53-moet.sh` builds only `Dockerfile.glm53-moet`, overlaying
this source tree's Python files onto the pinned vLLM image
(`vllm/vllm-openai:glm53-flash@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4`),
then adds the SM120 cubins under `/cubit-share`. The serve script uses TP2,
ModelOpt/Marlin, FP8 KV, the V2 runner, 32K context, one sequence, eager
execution, reasoning parser `glm45`, and tool parser `glm47`.

## Published container image

`ghcr.io/sepffuzzball/vllm-glm53-moet:latest` is published to GHCR by
`.github/workflows/publish-glm53-moet-image.yml`, which is also reachable from
GitHub's "Run workflow" button (`workflow_dispatch`) for manual builds. The
image is built for `linux/amd64` only (the included cubins are SM120-only).
`latest` tracks the default branch; tag pushes add a `v*` tag and every push
adds a `sha-<short>` tag.

The first GHCR publication is private by GitHub design; the workflow does not
change visibility. After the first publish, open
`https://github.com/users/sepffuzzball/packages/container/vllm-glm53-moet/settings`
once and choose Change visibility -> Public. `latest` is anonymously pullable
only after that one-time change; subsequent tags remain public.

To pull it:

```bash
docker pull ghcr.io/sepffuzzball/vllm-glm53-moet:latest
```

The previous script rebuilt the complete vLLM native image from source
(`docker/Dockerfile`, target `vllm-openai`) and could take hours. The current
build instead overlays Python onto the exact pinned base image, so it should be
mostly the base-image download plus a small overlay. If you already started an
old full source build (for example, it is at Docker build step 67 or 89), you
can stop it and just rerun `scripts/build-glm53-moet.sh`.

This overlay is ABI-safe only while the pinned base image and this source tree
are built from the same vLLM commit: the base image's compiled `.so`
extensions are reused as-is, so the overlaid Python must match their ABI. The
pinned base digest is recorded in `MOET_PROVENANCE.md`.

## Geometry and memory model

The language model has 45 layers. Layers 0-2 are dense and layers 3-44 are 42
sparse MoE layers. Each sparse layer has 288 routed experts, checkpoint top-k
8, one shared expert, hidden size 4096, and MoE intermediate size 2048. Under
TP2, routed gate/up uses K=4096 and down uses K=1024. The included cubins are
SM120-only.

The published checkpoint is about 181.3 GiB. Converting only routed experts to
the sign-symmetric W2 representation yields an estimated base footprint of
about 39.6 GiB per TP rank including block scales. Native attention, dense,
shared-expert, vision, router, and MTP weights remain at checkpoint precision.
The complete W2 base may therefore fit on each 96 GB GPU after conversion. The
cache profile instead limits the GPU W2 base cache to 32 GiB per rank, leaving
more headroom for KV cache and runtime workspaces.

The 2 GiB FP4 delta pool is a hot cache of original FP4 expert data used for
quality recovery. It is not a model delta published by the checkpoint author.

## Profiles

`MOET_PROFILE=cache` is the default. It enables a 32 GiB per-rank GPU expert
cache, a 32 GiB per-rank pinned host arena, persistent planes, and an NVMe pack
store. Replay is strict and the miss tolerance is zero.

```bash
MOET_PROFILE=cache scripts/serve-glm53-moet.sh
```

`MOET_PROFILE=resident` omits base-cache and store controls so the complete W2
base remains GPU resident. Use it first to separate quantization and kernel
correctness from cache behavior.

```bash
MOET_PROFILE=resident scripts/serve-glm53-moet.sh
```

Important overrides include `BASE_IMAGE` and `IMAGE` for the build script,
`IMAGE`, `MODEL_PATH`, `HF_CACHE`, `PLANES_CACHE`, `MOE_STORE`,
`BASE_CACHE_GB`, `BASE_RAM_GB`, `GPU_MEM_UTIL`, `MAX_BATCHED_TOKENS`,
`PORT`, and `ENFORCE_EAGER=0` for the serve script. Additional vLLM arguments
may be appended to the serve script.

Use at least 192 GiB of system RAM for the cache profile and first conversion.
Use a fast, bind-mounted ext4 or xfs filesystem, not container overlay storage,
with at least 500 GiB free for the checkpoint, W2 and FP4 packs, caches, and
operational headroom. First boot performs conversion and pack creation. Later
boots reuse packs only when checkpoint identity, revision, TP rank, geometry,
and quantizer metadata match.

## Phase-one boundary

The initial supported experiment is text-only, TP2, PP1, one sequence, 32K,
eager by default, no expert parallelism or EPLB, no MTP, and confidence gate
off. Pipeline parallel base caching fails closed. Multimodal requests, MTP,
compiled CUDA graphs, longer contexts, and higher concurrency require separate
validation.

Validate in this order:

1. Resident W2, delta disabled, eager short-generation smoke test.
2. Resident W2 plus FP4 delta and paired quality checks.
3. Strict 32 GiB base cache with replay and pack reboot tests.
4. CUDA graph execution and sustained decode tests.
5. MTP, longer context, and multimodal tests only after the baseline passes.

CPU-capable checks, after installing the project test environment, are:

```bash
.venv/bin/python -m pytest tests/model_executor/layers/test_moe_w2_mapped_host.py -q
.venv/bin/python -m pytest tests/model_executor/layers/test_moe_w2_persistence.py -q
```

GPU acceptance must additionally compare W2 and FP4 kernels against an
independent reference for K=4096 and K=1024, load the pinned checkpoint on TP2,
run short greedy generation, inspect strict replay convergence, and compare
quality against a native checkpoint baseline. None of those GPU/model checks
were run in this workspace.
