# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
MHA v4 Sol-Attn (arXiv 2607.24027): sparse dispatch, kernel accuracy, and compile parity.

Sol-Attn computes the KV blocks whose pooled proxy score clears a per-query-tile threshold exactly
and recovers the rest from pooled K/V instead of dropping them. The accuracy claim is therefore
comparative and is tested as such: against the same routed mask WITHOUT the correction term, which
is the A/B that isolates what the correction buys, and against dense attention.

The reference is evaluated on the mask the kernel was given, never on one it re-derives. Routing is
scale invariant in real arithmetic but not in fp32, so an oracle that re-routed would disagree about
blocks sitting within rounding distance of the threshold and the comparison would measure that
instead of the kernel.
"""

import math

import pytest
import torch
import torch._dynamo

from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    AttentionSparseMode,
    mha_v4,
    mha_v4_sol_attn,
    mha_v4_sol_attn_packed,
    native_fp8_format,
    quantize_fp8,
)
from aiter.ops.triton.attention.utils import (
    SOL_ATTN_TS_KV,
    SOL_ATTN_TS_QO,
    sol_attn_prepare,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_mha_common import attention_ref, sol_attn_ref

BETA = 0.5

pytestmark = pytest.mark.skipif(
    not arch_info.is_fp8_avail() or arch_info.get_arch() != "gfx950",
    reason="Sol-Attn has one kernel row, gfx950 FP8",
)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv, d=128, seed=0):
    """BF16 BSHD operands with a shared smooth positional component.

    Attention has to actually concentrate for routing to have anything to find: with independent
    noise the proxy scores are near uniform, every block looks equally useful, and a broken kernel
    would still land close to the reference.
    """
    torch.manual_seed(seed)
    walk = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda") / (seqlen_k**0.5)
    traj = walk.cumsum(dim=1)
    traj = traj / traj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    amp = 2.0 * (d**0.5)
    q_traj = (
        torch.nn.functional.interpolate(
            traj.permute(0, 2, 3, 1).reshape(batch * nhead_kv, d, seqlen_k),
            size=seqlen_q,
            mode="linear",
            align_corners=False,
        )
        .reshape(batch, nhead_kv, d, seqlen_q)
        .permute(0, 3, 1, 2)
    )
    q = torch.randn(batch, seqlen_q, nhead_q, d, device="cuda") + amp * (
        q_traj.repeat_interleave(nhead_q // nhead_kv, dim=2)
    )
    k = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda") + amp * traj
    v = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda")
    return (
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
    )


def _reference(q, k, v, beta=BETA, correction=True):
    """Quantize and route exactly as the entrypoint does, then evaluate the oracle on that mask."""
    q_quant, q_descale = quantize_fp8(q)
    k_quant, k_descale = quantize_fp8(k)
    v_quant, v_descale = quantize_fp8(v)
    routing = sol_attn_prepare(
        q_quant, k_quant, v_quant, beta, SOL_ATTN_TS_QO, SOL_ATTN_TS_KV
    )
    out, _ = sol_attn_ref(
        q_quant.float() * q_descale,
        k_quant.float() * k_descale,
        v_quant.float() * v_descale,
        routing["block_attn_mask"],
        routing["mean_k"].float() * k_descale,
        routing["mean_v"].float() * v_descale,
        SOL_ATTN_TS_QO,
        SOL_ATTN_TS_KV,
        correction=correction,
    )
    return out, routing


def _rel_error(actual, expected):
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp(min=1e-9)
    ).item()


def assert_launches_agree(actual, expected, what):
    """Two launches that must produce the same answer, allowing one known kernel defect through.

    The gfx950 FP8 forward kernels have an open nondeterminism, characterised in pyisa's
    ASM/fmha_sage_fwd/tools/SPARSE_FLAKE.md: given bitwise identical inputs, an occasional launch
    corrupts a single 32-row x 32-dim accumulator tile of one work item, always in output dims
    96-127 of rows 128-255 of one query tile. It is inherited from the shared sparse LUT walker,
    the dense kernel shows a weaker form of it, and reproducing it requires distinct kernel modules
    to be interleaved, which a test session does.

    So bitwise equality here fails roughly one session in five over a defect that lives in the code
    object rather than in this port, while a norm-based tolerance would hide a real regression
    behind averaging. Bounding the footprint does neither: anything systematic, such as a wrong
    stride, a mis-selected block or a dropped correction term, moves far more than one tile and
    is not confined to the last accumulator group.
    """
    if torch.equal(actual, expected):
        return
    where = (actual.float() != expected.float()).nonzero()
    tiles = " ".join(
        f"tile {int(t)} head {int(h)}"
        for t, h in {(int(r[1]) // SOL_ATTN_TS_QO, int(r[2])) for r in where}
    )
    detail = (
        f"{what}: {len(where)} elements differ across {tiles}, "
        f"dims {int(where[:, 3].min())}..{int(where[:, 3].max())}, "
        f"rows within tile {int((where[:, 1] % SOL_ATTN_TS_QO).min())}.."
        f"{int((where[:, 1] % SOL_ATTN_TS_QO).max())}, "
        f"max delta {(actual.float() - expected.float()).abs().max().item():.3e}"
    )
    assert len(where) <= 4 * 32 * 32, (
        f"{detail} (too widespread for the known kernel flake)"
    )
    assert int(where[:, 3].min()) >= 96, (
        f"{detail} (outside the known flake's dim group)"
    )


SHAPES = [
    (1, 4096, 4096, 8, 8),
    (1, 4096, 4096, 8, 2),
    (2, 1024, 1024, 4, 4),
    (1, 9419, 9419, 5, 5),
    (1, 640, 384, 4, 4),
]


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_matches_reference(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """The kernel reproduces the reference's exact + approximate mix on the same routing."""
    q, k, v = _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    expected, routing = _reference(q, k, v)
    actual = mha_v4_sol_attn(q, k, v, BETA)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == q.shape
    assert torch.isfinite(actual.float()).all(), "kernel produced non-finite output"
    selected = routing["block_attn_mask"].float().mean().item()
    # fp8 operands and a bf16 epilogue put the floor here; the correction term is the thing under
    # test, and it moves the error by far more than this tolerance.
    assert _rel_error(actual, expected) < 6e-2, (
        f"selected fraction {selected:.3f}, rel error {_rel_error(actual, expected):.4f}"
    )


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES[:3])
def test_correction_beats_dropping_the_same_blocks(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """Recovering skipped blocks from pooled K/V must beat dropping them, on identical routing.

    Both sides are the reference, so this isolates the correction term from the kernel's fp8 and
    BF16 rounding. beta is high enough that the skipped blocks still carry mass: at the default
    beta the selected blocks already capture nearly all of it and the correction is a no-op, which
    is a property of the routing, not a defect.
    """
    beta = 1.5
    q, k, v = _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    dense, _, _ = attention_ref(q.float(), k.float(), v.float(), upcast=True)
    corrected, routing = _reference(q, k, v, beta=beta, correction=True)
    dropped, _ = _reference(q, k, v, beta=beta, correction=False)

    selected = routing["block_attn_mask"].float().mean().item()
    assert selected < 0.5, "routing selected too much for the correction to matter"
    assert _rel_error(corrected, dense) < _rel_error(dropped, dense), (
        f"correction did not help: {_rel_error(corrected, dense):.4f} vs "
        f"{_rel_error(dropped, dense):.4f} at selected fraction {selected:.3f}"
    )


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES[:3])
def test_kernel_runs_the_approximate_branch(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """The kernel must land on the corrected reference, not on the same mask with blocks dropped.

    A kernel that ignored the pooled operands would still look accurate against a loose dense
    tolerance at moderate density. This separates the two hypotheses directly.
    """
    beta = 1.5
    q, k, v = _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    actual = mha_v4_sol_attn(q, k, v, beta)
    corrected, _ = _reference(q, k, v, beta=beta, correction=True)
    dropped, _ = _reference(q, k, v, beta=beta, correction=False)

    to_corrected = _rel_error(actual, corrected)
    to_dropped = _rel_error(actual, dropped)
    assert to_corrected < 0.5 * to_dropped, (
        f"kernel sits {to_corrected:.4f} from the corrected reference and {to_dropped:.4f} from "
        "the dropped one; the approximate branch does not look active"
    )


@pytest.mark.parametrize(
    "batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", [SHAPES[0], SHAPES[3]]
)
def test_tracks_dense_fp8_at_production_density(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """At the default threshold, skipping most of the KV blocks must cost little against dense FP8.

    Dense FP8 is the right yardstick rather than BF16 attention: per-tensor FP8 quantization is
    worth ~17% relative error on this data by itself, which would swamp what sparsity costs.
    """
    q, k, v = _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    fp8 = native_fp8_format()
    dense_fp8 = mha_v4(q, k, v, fp8, fp8, fp8)
    actual = mha_v4_sol_attn(q, k, v, BETA)
    _, routing = _reference(q, k, v)

    selected = routing["block_attn_mask"].float().mean().item()
    assert selected < 0.5, (
        f"expected the default threshold to be selective, got {selected:.3f}"
    )
    assert _rel_error(actual, dense_fp8) < 5e-2, (
        f"selected fraction {selected:.3f}, rel error against dense FP8 "
        f"{_rel_error(actual, dense_fp8):.4f}"
    )


@pytest.mark.parametrize("beta", [-1.0, 0.0, 0.5, 1.5, 3.0])
def test_kernel_follows_the_reference_across_beta(beta):
    """Whatever the threshold selects, the kernel must compute that mix.

    Only agreement with the reference is asserted, not accuracy against dense attention: at a high
    beta the answer legitimately diverges from dense, because that is what asking for 6% of the KV
    blocks means. What the production threshold costs is measured separately.
    """
    q, k, v = _inputs(1, 2048, 2048, 8, 8)
    actual = mha_v4_sol_attn(q, k, v, beta)
    expected, routing = _reference(q, k, v, beta=beta)

    selected = routing["block_attn_mask"].float().mean().item()
    assert torch.isfinite(actual.float()).all()
    assert _rel_error(actual, expected) < 6e-2, f"beta={beta}, selected={selected:.3f}"


def test_sparsity_actually_varies_with_beta():
    q, k, v = _inputs(1, 2048, 2048, 8, 8)
    fractions = [
        sol_attn_prepare(
            *[t[0] for t in (quantize_fp8(q), quantize_fp8(k), quantize_fp8(v))],
            beta,
            SOL_ATTN_TS_QO,
            SOL_ATTN_TS_KV,
        )["block_attn_mask"]
        .float()
        .mean()
        .item()
        for beta in (-1.0, 0.5, 3.0)
    ]
    assert fractions[0] > fractions[1] > fractions[2], fractions


def test_out_is_caller_allocated_and_returned():
    """The launch mutates the caller's buffer, which is what keeps the compiled fake trivial."""
    q, k, v = _inputs(1, 1024, 1024, 4, 4)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    returned = mha_v4_sol_attn(q, k, v, BETA, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.isfinite(out.float()).all()


def test_packed_api_matches_raw_api():
    q, k, v = _inputs(1, 2048, 2048, 8, 8)
    fp8 = native_fp8_format()
    q_quant, q_descale = quantize_fp8(q)
    k_quant, k_descale = quantize_fp8(k)
    v_quant, v_descale = quantize_fp8(v)
    routing = sol_attn_prepare(
        q_quant, k_quant, v_quant, BETA, SOL_ATTN_TS_QO, SOL_ATTN_TS_KV
    )
    packed = mha_v4_sol_attn_packed(
        q_quant,
        k_quant,
        v_quant,
        q_descale,
        k_descale,
        v_descale,
        routing["mean_k"],
        routing["mean_v"],
        routing["kv_block_indices"],
        routing["lut_start"],
        routing["lut_count"],
        routing["block_bitmap"],
        fp8,
        fp8,
        fp8,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        fp8,
        AttentionScaleMode.F32_PER_TENSOR,
    )
    assert_launches_agree(packed, mha_v4_sol_attn(q, k, v, BETA), "packed against raw")


def test_softmax_scale_is_honoured():
    q, k, v = _inputs(1, 1024, 1024, 4, 4)
    default = mha_v4_sol_attn(q, k, v, BETA)
    explicit = mha_v4_sol_attn(q, k, v, BETA, softmax_scale=1.0 / math.sqrt(128))
    assert_launches_agree(explicit, default, "explicit default scale")
    other = mha_v4_sol_attn(q, k, v, BETA, softmax_scale=0.5 / math.sqrt(128))
    assert _rel_error(other, default) > 1e-3, (
        "the softmax scale did not reach the kernel"
    )


def test_repeated_calls_agree():
    q, k, v = _inputs(1, 2048, 2048, 8, 8)
    first = mha_v4_sol_attn(q, k, v, BETA)
    second = mha_v4_sol_attn(q, k, v, BETA)
    assert_launches_agree(second, first, "a repeated call")


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES[:4])
def test_compiles_fullgraph_and_matches_eager(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """No custom op on the caller's side: fullgraph tracing goes straight through the entrypoint."""
    q, k, v = _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    eager = mha_v4_sol_attn(q, k, v, BETA)
    compiled = torch.compile(mha_v4_sol_attn, fullgraph=True, dynamic=False)(
        q, k, v, BETA
    )
    assert_launches_agree(compiled, eager, "compiled against eager")


def test_no_graph_breaks():
    q, k, v = _inputs(1, 4096, 4096, 8, 8)
    explained = torch._dynamo.explain(mha_v4_sol_attn)(q, k, v, BETA)
    assert explained.graph_break_count == 0, (
        f"graph breaks: {[str(r) for r in explained.break_reasons]}"
    )


def test_compiled_output_survives_allocator_churn():
    """A downstream consumer plus reallocation, which is where a wrong output layout shows up."""
    q, k, v = _inputs(1, 2048, 2048, 8, 8)
    compiled = torch.compile(mha_v4_sol_attn, fullgraph=True, dynamic=False)
    expected = mha_v4_sol_attn(q, k, v, BETA)
    for _ in range(4):
        churn = torch.empty(1 << 22, device="cuda", dtype=torch.bfloat16)
        out = compiled(q, k, v, BETA)
        consumed = (out.float() * 2.0).sum(dim=-1)
        assert torch.isfinite(consumed).all()
        assert_launches_agree(out, expected, "compiled output after reallocation")
        del churn


def test_dispatch_is_explicit_about_unsupported_requests():
    q, k, v = _inputs(1, 1024, 1024, 4, 4)
    with pytest.raises(NotImplementedError, match="one kernel row"):
        mha_v4_sol_attn(
            q,
            k,
            v,
            BETA,
            q_format=AttentionFormat.MXFP4,
            k_format=AttentionFormat.MXFP4,
            v_format=AttentionFormat.MXFP4,
        )
    with pytest.raises(NotImplementedError, match="LSE"):
        mha_v4_sol_attn(q, k, v, BETA, return_lse=True)
    with pytest.raises(ValueError, match="head dimension 128"):
        mha_v4_sol_attn(
            q[..., :64].contiguous(),
            k[..., :64].contiguous(),
            v[..., :64].contiguous(),
            BETA,
        )
    with pytest.raises(ValueError, match="BF16"):
        mha_v4_sol_attn(q.float(), k.float(), v.float(), BETA)


def test_non_power_of_two_gqa_is_refused():
    """The kernel shifts to find a KV head, so an unsupported ratio must fail, not misroute."""
    q, k, v = _inputs(1, 1024, 1024, 6, 2)  # ratio 3
    with pytest.raises(ValueError, match="power-of-two"):
        mha_v4_sol_attn(q, k, v, BETA)

    # Not a multiple at all, so the inputs cannot come from the GQA-shaped generator.
    def heads(h):
        return torch.randn(1, 1024, h, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="multiple of KV heads"):
        mha_v4_sol_attn(heads(6), heads(4), heads(4), BETA)


def test_sparse_mode_selects_the_row():
    """A dense request must never reach the sparse kernel, and vice versa."""
    from aiter.ops.mha_v4 import _mha_v4_fwd_sparse_launch

    q, k, v = _inputs(1, 1024, 1024, 4, 4)
    fp8 = native_fp8_format()
    q_quant, q_descale = quantize_fp8(q)
    k_quant, k_descale = quantize_fp8(k)
    v_quant, v_descale = quantize_fp8(v)
    routing = sol_attn_prepare(
        q_quant, k_quant, v_quant, BETA, SOL_ATTN_TS_QO, SOL_ATTN_TS_KV
    )
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    with pytest.raises(RuntimeError, match="pooled correction"):
        _mha_v4_fwd_sparse_launch(
            q_quant,
            k_quant,
            v_quant,
            q_descale,
            k_descale,
            v_descale,
            routing["mean_k"],
            routing["mean_v"],
            routing["kv_block_indices"],
            routing["lut_start"],
            routing["lut_count"],
            routing["block_bitmap"],
            out,
            int(fp8),
            int(fp8),
            int(fp8),
            int(AttentionScaleMode.F32_PER_TENSOR),
            int(AttentionScaleMode.F32_PER_TENSOR),
            int(AttentionScaleMode.F32_PER_TENSOR),
            int(fp8),
            int(AttentionScaleMode.F32_PER_TENSOR),
            int(AttentionSparseMode.BLOCK_LUT),
            128**-0.5,
        )
