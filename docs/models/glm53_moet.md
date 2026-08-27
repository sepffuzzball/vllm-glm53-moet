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
`IMAGE`, `MODEL_PATH`, `REVISION`, `MODEL_IDENTITY`, `HF_CACHE`,
`PLANES_CACHE`, `MOE_STORE`, `BASE_CACHE_GB`, `BASE_RAM_GB`,
`GPU_MEM_UTIL`, `MAX_BATCHED_TOKENS`, `PORT`, and `ENFORCE_EAGER=0` for
the serve script. Additional vLLM arguments may be appended to the serve
script.

Use at least 192 GiB of system RAM for the cache profile and first conversion.
Use a fast, bind-mounted ext4 or xfs filesystem, not container overlay storage,
with at least 500 GiB free for the checkpoint, W2 and FP4 packs, caches, and
operational headroom. First boot performs conversion and pack creation. Later
boots reuse packs only when checkpoint identity, revision, TP rank, geometry,
and quantizer metadata match.

## Serving a local model directory

By default the script serves the pinned remote repository
(`dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4` at revision
`835b767e640aeaace97bd9d8b6d4ddecd9d8e9d4`). To run from a model that is
already downloaded on the host, set `MODEL_PATH` to its directory. The script
resolves the path, bind-mounts it read-only at `/model` inside the container,
passes `/model` as the model argument, and omits `--revision` (a revision
only applies to remote repositories):

```bash
MODEL_PATH=/home/hal/models/GLM-5.3-Flash-NVFP4a MODEL_IDENTITY=glm53-nvfp4a-v1 scripts/serve-glm53-moet.sh
```

The mount is read-only (`:ro`), so the container never mutates your host
files, and paths containing spaces are passed as a single argument. Any
`MODEL_PATH` that names an existing host directory is treated as a local
model, including a bare relative path such as `models/glm`. A `MODEL_PATH`
beginning with `/`, `./`, `../`, or `~` that does not exist on the host
fails before `docker run` is invoked; any other value must be a valid one-
or two-component Hugging Face repo ID.

Passing a host path directly (without a mount) does not work: the container
does not contain `/home/hal/models/GLM-5.3-Flash-NVFP4a`, so vLLM raises
`HFValidationError` ("does not exist"). The wrapper's bind mount is what
makes the path visible in-container.

### Direct image usage

Without the wrapper, mount the model and pass the in-container path yourself:

```bash
docker run --rm --gpus all --ipc=host --shm-size 64g \
  -p 127.0.0.1:8000:8000 \
  -v /home/hal/models/GLM-5.3-Flash-NVFP4a:/model:ro \
  -e VLLM_MOE_W2_CKPT_ID=glm53-nvfp4a-v1 \
  ghcr.io/sepffuzzball/vllm-glm53-moet:latest \
  /model --tensor-parallel-size 2 --quantization modelopt --moe-backend marlin \
  --kv-cache-dtype fp8 --max-model-len 32768 --max-num-seqs 1 \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
```

### HF cache snapshots

If the model directory lives inside the HF cache (for example
`$HF_CACHE/hub/models--dealignai--GLM-5.3-Flash-ABLITERATED-NVFP4/snapshots/<commit>`),
the wrapper instead mounts the whole cache read-only at `/model-hf-cache` and
passes the snapshot's path under it. Snapshot entries are symlinks into
`blobs/`; mounting the entire cache keeps those symlinks valid inside the
container.

### Checkpoint identity

`VLLM_MOE_W2_CKPT_ID` names the persisted MoET pack. It is set only for
local model directories:

- Local HF snapshot whose final directory name is a 40-hex commit: that
  commit, unless `MODEL_IDENTITY` is set.
- Any other local directory: `MODEL_IDENTITY` if set; otherwise the
  variable is omitted and the runtime derives a local metadata
  fingerprint on first load.

Remote repositories do not set `VLLM_MOE_W2_CKPT_ID`: the runtime uses
Hugging Face's resolved commit hash (from the `--revision` still passed on
the command line) as the checkpoint identity, so persistent pack reuse keys
on the exact snapshot Hugging Face actually resolved.

The wrapper warns when a local path is used without a usable identity (no
`MODEL_IDENTITY` and no 40-hex snapshot commit). Set `MODEL_IDENTITY` to an
immutable artifact ID (for example `glm53-nvfp4a-v1`) so persistent pack
reuse keys on a stable value; the pinned remote revision is never reused for
a local directory.

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
