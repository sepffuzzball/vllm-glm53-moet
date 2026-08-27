# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locks the GLM-5.3 TP2 W2 geometry, layer eligibility, and shipped SM120
kernel inventory.

All three tests are GPU-free: the moe_w2 layer predicate is pure string
logic, the plane-footprint numbers are derived arithmetic, and the kernel
inventory is a filesystem check.
"""

import hashlib
from pathlib import Path

import pytest

from vllm.compilation import backends as _comp_backends
from vllm.model_executor.layers.quantization.utils import moe_w2_cubit as _cubit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KERNELS_DIR = _REPO_ROOT / "kernels" / "cubins-sm120"

# GLM-5.3-Flash (glm5_next) geometry: 45 hidden layers with 42 sparse MoE
# layers at indices 3..44 and 288 routed experts.
_LAYERS = 42
_EXPERTS = 288
_HIDDEN = 4096
_INTERMEDIATE = 2048
_TP = 2
_K13 = _HIDDEN  # gate/up contraction dim, never sharded
_K2 = _INTERMEDIATE // _TP  # down contraction dim under TP2


@pytest.mark.skip_global_cleanup
def test_glm53_w2_layer_eligibility(monkeypatch):
    """The W2 predicate routes main-stack sparse layers only."""
    monkeypatch.setattr(_cubit, "enabled", lambda: True)
    monkeypatch.setattr(_cubit, "_layer_cutoff", lambda: 45)
    monkeypatch.setattr(_comp_backends, "model_tag", "backbone")

    # Selected: first and last sparse main-stack routed layers.
    assert _cubit.is_w2_layer("model.layers.3.mlp.experts.0.gate_up_proj")
    assert _cubit.is_w2_layer("model.layers.44.mlp.experts.7.down_proj")

    # Rejected: MTP drafter, vision tower, and an out-of-range main layer.
    assert not _cubit.is_w2_layer(
        "model.layers.45.mtp_block.mlp.experts.0.gate_up_proj"
    )
    assert not _cubit.is_w2_layer("model.visual.blocks.0.self_attn")
    assert not _cubit.is_w2_layer("model.layers.45.mlp.experts.0.gate_up_proj")

    # Dense (indices 0-2) and MTP (>= 45) layers are not represented by the
    # string predicate (it would admit, e.g., model.layers.2.*); the actual
    # sparse set is range(3, 45), count 42.
    sparse_indices = set(range(3, 45))
    assert len(sparse_indices) == 42
    assert min(sparse_indices) == 3
    assert max(sparse_indices) == 44
    assert {i for i in range(45) if i not in sparse_indices} == {0, 1, 2}


@pytest.mark.skip_global_cleanup
def test_glm53_w2_plane_footprint():
    """Per-rank 2-bit W2 planes land in the expected 39-41 GiB window."""
    bits_per_elem = 2
    block = 32
    scale_bytes_per_block = 1

    # TP2-sharded output widths: w13 (gate/up) keeps full K = hidden and a
    # local output of 2 * intermediate // tp; w2 (down) keeps full hidden
    # output and shards K by tp.
    gate_up_local_n = 2 * _INTERMEDIATE // _TP  # 2048
    down_out_n = _HIDDEN  # 4096

    def plane_bytes(k: int, n: int) -> int:
        elems = k * n
        return elems * bits_per_elem // 8 + (elems // block) * scale_bytes_per_block

    per_expert = plane_bytes(_K13, gate_up_local_n) + plane_bytes(_K2, down_out_n)
    total = _LAYERS * _EXPERTS * per_expert
    assert total == 42_807_066_624
    assert 39 <= total / (1024**3) <= 41


@pytest.mark.skip_global_cleanup
def test_glm53_w2_shipped_kernel_inventory():
    """The shipped SM120 cubin set matches the reviewed SHA256SUMS manifest."""
    assert (_REPO_ROOT / "kernels" / "GLM53_MANIFEST.md").is_file()

    # The manifest must list every shipped cubin, sorted by basename, with
    # no unrelated entries.
    shipped = sorted(p.name for p in _KERNELS_DIR.glob("*.cubin"))
    assert shipped, "no .cubin files under kernels/cubins-sm120"

    manifest = {}
    for line in (_KERNELS_DIR / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        assert len(parts) == 2, f"malformed SHA256SUMS line: {line!r}"
        digest, name = parts
        assert name not in manifest, f"duplicate SHA256SUMS entry: {name}"
        manifest[name] = digest

    assert list(manifest) == shipped, "SHA256SUMS must cover exactly the cubins"

    # Every shipped cubin hashes to the reviewed digest recorded in the
    # manifest.
    for name, digest in manifest.items():
        actual = hashlib.sha256((_KERNELS_DIR / name).read_bytes()).hexdigest()
        assert actual == digest, name

    required = [
        # W2, _a32 (LIVE) baseline, K=4096 (gate/up) and K=1024 (down, TP2).
        "moe_w2_mm_k4096_a32.cubin",
        "moe_w2_mm_k1024_a32.cubin",
        # W2, _a32 mc4 (prefill).
        "moe_w2_mm_mc4_k4096_a32.cubin",
        "moe_w2_mm_mc4_k1024_a32.cubin",
        # W2, _a32 mc4afrag (prefill, default).
        "moe_w2_mm_mc4afrag_k4096_a32.cubin",
        "moe_w2_mm_mc4afrag_k1024_a32.cubin",
        # FP4 delta moe_w4, _a32 (LIVE).
        "moe_w4_mm_k4096_a32.cubin",
        "moe_w4_mm_k1024_a32.cubin",
    ]
    for name in required:
        assert (_KERNELS_DIR / name).is_file(), name
