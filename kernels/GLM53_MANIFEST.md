# GLM-5.3 MoE W2 runtime cubin manifest (SM120 / sm_120)

This is a GLM-5.3-specific manifest for the MoE W2 runtime cubin set copied
into `kernels/cubins-sm120/`. It is a separate document from the source
repo's `kernels/MANIFEST.md` (which covers the full upstream family and is
left untouched here).

## Source reference (exact)

- Remote: `kacper-moet` -> https://github.com/kacper-daftcode/vLLM-Moet.git
- Branch: `main`
- Source ref (commit): `de0eb87e2d70e03dffcf6737cc51f36cf4323e16`
- Source path: `kernels/cubins-sm120/` (files copied from this tree
  byte-for-byte; filenames are byte-identical to the source artifacts)

## SM120-only constraint

All cubins are assembled for **SM120 only** (sm_120, Blackwell). They will
not load on any other compute capability. Hand-written SASS assembled by
`cubit` (pinned @ `5912400`). Do not use these binaries on non-SM120
hardware.

## Selection rule

Copied files are exactly the tracked files under `kernels/cubins-sm120` in
the source ref whose basename belongs to the `moe_w2`, `moe_w4`, or
`moe_w4q` families **and** contains `_k4096` or `_k1024`, including
baseline, `a32`, `mc2`, `mc4`, and `mc4afrag` variants where present.
No other K geometries, no other families (`moe_w4s`, `exl3-wave-m8`,
`mla_prefill_state`) are included.

## Geometry purpose (GLM-5.3, TP2)

- **gate/up (w13, hidden H): K = 4096, never sharded.** The `*_k4096`
  cubins serve the gate/up contraction at full width on every rank.
- **down (w2, intermediate I): K = 1024 under TP2.** The `*_k1024`
  cubins serve the down contraction sharded per rank at TP2 (K/2). TP4
  would need K = 512 (`*_k512`) cubins, which are NOT copied here.

Variant meaning (matches the source kernel families):

- `moe_w2_mm_*` : 2-bit sign-symmetric `{-4,-1,1,4}` planes + UE8M0
  block-32 scales, decoded to QMMA.SF tensor-core GEMM.
- `moe_w4_mm_*` : FP4 (e2m1) hot-expert delta GEMM.
- `moe_w4q_mm_*` : split FP4 (radix-5 "quintal" refinement plane) GEMM,
  opt-in via `VLLM_MOE_W2_DELTA_SPLIT=1` (default off).
- `_a32` suffix : per-32-group flexible A scale format (the LIVE set the
  loader opens; regcount 64, 4 CTA/SM). Non-suffixed files are the
  retired a128 files retained for in-flight sessions.
- `mc2` / `mc4` : MC=2 / MC=4 (prefill) variants of `moe_w2_mm`. `mc2`
  is retired for good (not regenerated in `_a32`) but the source still
  tracks the files; copied here for completeness.
- `mc4afrag` : MC=4 prefill with fragment-major A repack (default ON via
  `VLLM_MOE_W2_AFRAG`; set 0 to fall back to mc4). Bit-identical output
  vs mc4.

## Copied files (22), blob hashes, sizes

Blob hashes are the git blob SHA-1 of each file at the source ref above.

### moe_w2, baseline (a128, retired) and a32 (LIVE), K=4096 (gate/up)

| cubin | blob hash | size (bytes) | purpose |
|---|---|---|---|
| `moe_w2_mm_k4096.cubin` | `a4f57fd4296d8d634aea67be26f2f8a0ce8b9813` | 21776 | 2-bit MoE GEMM, MC=1 (decode), K=4096 (gate/up) |
| `moe_w2_mm_k4096_a32.cubin` | `959d2d0770429ac73034c895c52b9d1ec18b17b7` | 23488 | 2-bit MoE GEMM, MC=1 (decode), K=4096, a32 (LIVE) |
| `moe_w2_mm_mc2_k4096.cubin` | `9461b8d44778582a181e38799110fb7767c162b6` | 27088 | 2-bit MoE GEMM, MC=2 (prefill), K=4096 (retired) |
| `moe_w2_mm_mc4_k4096.cubin` | `bbdd3d2dfcaad5730b9a63df7ce4b15a81dfca4a` | 22896 | 2-bit MoE GEMM, MC=4 (prefill), K=4096 |
| `moe_w2_mm_mc4_k4096_a32.cubin` | `f74cd27d5e696290102f6f7865856a54d16e7f2a` | 26368 | 2-bit MoE GEMM, MC=4 (prefill), K=4096, a32 (LIVE) |
| `moe_w2_mm_mc4afrag_k4096.cubin` | `106a015e6f9b51bdd286d5561708d011076a8eb1` | 22112 | 2-bit MoE GEMM, MC=4 AFRAG (prefill), K=4096 |
| `moe_w2_mm_mc4afrag_k4096_a32.cubin` | `66ec17a48d382710f04f9ad76f13b0b99abcfab2` | 25584 | 2-bit MoE GEMM, MC=4 AFRAG (prefill), K=4096, a32 (LIVE) |

### moe_w2, baseline and a32, K=1024 (down under TP2)

| cubin | blob hash | size (bytes) | purpose |
|---|---|---|---|
| `moe_w2_mm_k1024.cubin` | `9dbdc0a39bb48421e9ccf0eb4a27a33a688cf53a` | 11264 | 2-bit MoE GEMM, MC=1 (decode), K=1024 (down, TP2) |
| `moe_w2_mm_k1024_a32.cubin` | `66a25c16a49100ef0f00a8b5d8c3529ec0097598` | 11680 | 2-bit MoE GEMM, MC=1 (decode), K=1024, a32 (LIVE) |
| `moe_w2_mm_mc2_k1024.cubin` | `4511b0c8f904e02290a038c66bb42c79e717c5df` | 13456 | 2-bit MoE GEMM, MC=2 (prefill), K=1024 (retired) |
| `moe_w2_mm_mc4_k1024.cubin` | `4367d1187ae57536fbf311ceced8f22f75a36b47` | 11568 | 2-bit MoE GEMM, MC=4 (prefill), K=1024 |
| `moe_w2_mm_mc4_k1024_a32.cubin` | `eb058f36fde4d6325da9e1f725368faa2db90757` | 12448 | 2-bit MoE GEMM, MC=4 (prefill), K=1024, a32 (LIVE) |
| `moe_w2_mm_mc4afrag_k1024.cubin` | `a89ed958ef3837f57f72a09f789aa66a93e7df63` | 11360 | 2-bit MoE GEMM, MC=4 AFRAG (prefill), K=1024 |
| `moe_w2_mm_mc4afrag_k1024_a32.cubin` | `5bbd0849b359fbf7c4792a579ff21babe501e6ec` | 12240 | 2-bit MoE GEMM, MC=4 AFRAG (prefill), K=1024, a32 (LIVE) |

### moe_w4 (FP4 delta), baseline and a32

| cubin | blob hash | size (bytes) | purpose |
|---|---|---|---|
| `moe_w4_mm_k4096.cubin` | `79ffd2a03e680b9fb06fc8f04ce167cec9db6875` | 25040 | FP4 delta GEMM, K=4096 (gate/up) |
| `moe_w4_mm_k4096_a32.cubin` | `1f236007f23814dff05125706f0c56c559a9856a` | 27360 | FP4 delta GEMM, K=4096, a32 (LIVE) |
| `moe_w4_mm_k1024.cubin` | `0f95591a2a4e02612dd3f39f085231fb4d3724b6` | 12128 | FP4 delta GEMM, K=1024 (down, TP2) |
| `moe_w4_mm_k1024_a32.cubin` | `a6e74cd45d955778cc432ea6c763c6858919b6c0` | 12480 | FP4 delta GEMM, K=1024, a32 (LIVE) |

### moe_w4q (split FP4, opt-in), baseline and a32

| cubin | blob hash | size (bytes) | purpose |
|---|---|---|---|
| `moe_w4q_mm_k4096.cubin` | `dc940d9a6c1f986a10abc12210acea3fc3df238c` | 42912 | quintal split FP4 GEMM, K=4096 (gate/up) |
| `moe_w4q_mm_k4096_a32.cubin` | `79c7ef34952d5257f7ef00732aeb687fc6d622e4` | 44560 | quintal split FP4 GEMM, K=4096, a32 (LIVE) |
| `moe_w4q_mm_k1024.cubin` | `e39f0444d60afde6ce1bfbe00959dd7f4a500bcf` | 16560 | quintal split FP4 GEMM, K=1024 (down, TP2) |
| `moe_w4q_mm_k1024_a32.cubin` | `32155ab43d86c0e7dedb5cf0aabb6e84edf4db7e` | 16912 | quintal split FP4 GEMM, K=1024, a32 (LIVE) |

## Integrity

Each file was copied byte-for-byte from the source tree via
`git show kacper-moet/main:<path>` and verified by SHA-256 against the
source blob (all 22 match). 22 files, 451280 bytes total. The blob hashes
above let anyone re-verify against `de0eb87e2d70e03dffcf6737cc51f36cf4323e16`.
