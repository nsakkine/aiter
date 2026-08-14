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

Accuracy is asserted against the kernel's own FP8 arithmetic floor rather than a fixed constant. At
full density the oracle reduces exactly to dense attention over the dequantized FP8 operands, so the
residual there is what FP8 costs by itself; comparing the sparse residual against it asserts that
sparsity and the correction add nothing measurable, rather than asserting that some total happens to
land under a round number. That floor depends on both the shape and the softmax scale, so it is
measured per case instead of written down.

Everything that can be exact is exact. The kernel is deterministic, and at full density it agrees
with the already validated dense FP8 row bit for bit, which pins the LUT walker, the GQA head
mapping, the operand strides and the epilogue with no tolerance at all.
"""

import csv
import math
from pathlib import Path

import pytest
import torch
import torch._dynamo

import aiter
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

# A threshold this far below the mean puts every KV block in the exact branch, which is how the FP8
# arithmetic floor is measured: the correction branch is then entirely masked out on both sides.
FULL_DENSITY_BETA = -1e9

# How far above that floor the routed cases are allowed to sit. Measured ratios span 0.91 to 1.00
# across the shapes and betas exercised here, on both MHA and GQA, so sparsity costs nothing beyond
# FP8 itself and this is headroom rather than budget.
FLOOR_MARGIN = 1.10

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


def _reference(q, k, v, beta=BETA, correction=True, softmax_scale=None):
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
        softmax_scale=softmax_scale,
        correction=correction,
    )
    return out, routing


def _rel_error(actual, expected):
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp(min=1e-9)
    ).item()


def _fp8_arithmetic_floor(q, k, v, softmax_scale=None):
    """What separates the kernel from the oracle when no block is approximated.

    Routing everything into the exact branch masks the correction out of both sides, and the oracle's
    exact branch is plain dense attention over the dequantized FP8 operands, computed in fp32. So the
    residual here is entirely the kernel's own arithmetic: FP8 MFMA operands, the softmax, the FP8
    probability round-trip into the PV product, and the BF16 epilogue.

    Routed cases are compared against this instead of against a constant. It depends on the shape and
    on the softmax scale -- both change how many columns the FP8 probability noise averages over --
    so it has to be measured for the case at hand, and callers must pass the same scale they test.
    """
    reference, _ = _reference(
        q, k, v, beta=FULL_DENSITY_BETA, softmax_scale=softmax_scale
    )
    actual = mha_v4_sol_attn(q, k, v, FULL_DENSITY_BETA, softmax_scale=softmax_scale)
    return _rel_error(actual, reference)


def assert_launches_agree(actual, expected, what):
    """Two launches of the same computation must agree bitwise.

    The kernel is deterministic: one code object over identical inputs writes identical bytes,
    whatever else the process happens to be doing. Nothing here is tolerated, and the report below
    exists only to localise a failure -- which accumulator group, which query tile, which head --
    because that is what distinguishes a wrong stride from a wrong block selection when one of
    these does fail.
    """
    if torch.equal(actual, expected):
        return
    where = (actual.float() != expected.float()).nonzero()
    tiles = " ".join(
        f"tile {int(t)} head {int(h)}"
        for t, h in sorted({(int(r[1]) // SOL_ATTN_TS_QO, int(r[2])) for r in where})
    )
    raise AssertionError(
        f"{what}: {len(where)} elements differ across {tiles}, "
        f"dims {int(where[:, 3].min())}..{int(where[:, 3].max())}, "
        f"rows within tile {int((where[:, 1] % SOL_ATTN_TS_QO).min())}.."
        f"{int((where[:, 1] % SOL_ATTN_TS_QO).max())}, "
        f"max delta {(actual.float() - expected.float()).abs().max().item():.3e}"
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
    """The kernel reproduces the reference's exact + approximate mix on the same routing.

    The bound is the case's own FP8 arithmetic floor, so this fails if routing the workload sparsely
    costs anything the same kernel does not already cost at full density.
    """
    q, k, v = _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    expected, routing = _reference(q, k, v)
    actual = mha_v4_sol_attn(q, k, v, BETA)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == q.shape
    assert torch.isfinite(actual.float()).all(), "kernel produced non-finite output"
    error = _rel_error(actual, expected)
    floor = _fp8_arithmetic_floor(q, k, v)
    selected = routing["block_attn_mask"].float().mean().item()
    assert error <= floor * FLOOR_MARGIN, (
        f"selected fraction {selected:.3f}, rel error {error:.5f} against an FP8 floor of "
        f"{floor:.5f} (ratio {error / floor:.3f}), so sparsity is costing accuracy"
    )


MHA_SHAPES = [shape for shape in SHAPES if shape[3] == shape[4]]


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", MHA_SHAPES)
@pytest.mark.parametrize("softmax_scale", [None, 0.5 / math.sqrt(128), 1.0])
def test_full_density_matches_the_dense_row_bitwise(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv, softmax_scale
):
    """Routed to full density, the sparse row must reproduce the dense FP8 row byte for byte.

    This is the strongest statement available about the sparse kernel and it needs no tolerance. With
    every block selected the correction branch is inert, so the sparse row is doing exactly the dense
    row's work through a different code path: the LUT walker instead of a counted loop, a GQA head
    shift, its own operand strides and its own epilogue. Any error in that machinery -- an off-by-one
    block index, a stride in elements where bytes were meant, a dropped tail -- breaks equality here
    while leaving a norm-based accuracy check comfortably green.

    Restricted to matching head counts because the dense v4 row is MHA only; GQA is covered by the
    floor-relative tests above.
    """
    q, k, v = _inputs(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    fp8 = native_fp8_format()
    sparse = mha_v4_sol_attn(q, k, v, FULL_DENSITY_BETA, softmax_scale=softmax_scale)
    dense = mha_v4(q, k, v, fp8, fp8, fp8, softmax_scale=softmax_scale)
    assert_launches_agree(sparse, dense, "sparse at full density against the dense row")


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
    error = _rel_error(actual, dense_fp8)
    assert selected < 0.5, (
        f"expected the default threshold to be selective, got {selected:.3f}"
    )
    # Measured 0.0267 at 4096 and 0.0281 at 9419, at roughly 36% density. The bound is set close to
    # those so that a regression in the correction term has nowhere to hide; it is not a claim about
    # what Sol-Attn costs on arbitrary data, which depends on how concentrated the attention is.
    assert error < 3.5e-2, (
        f"selected fraction {selected:.3f}, rel error against dense FP8 {error:.5f}"
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
    error = _rel_error(actual, expected)
    floor = _fp8_arithmetic_floor(q, k, v)
    assert torch.isfinite(actual.float()).all()
    assert error <= floor * FLOOR_MARGIN, (
        f"beta={beta}, selected={selected:.3f}, rel error {error:.5f} against an FP8 floor of "
        f"{floor:.5f} (ratio {error / floor:.3f})"
    )


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


def test_out_is_caller_allocated_and_fully_written():
    """The launch mutates the caller's buffer, which is what keeps the compiled fake trivial.

    Seeded with NaN rather than left uninitialized, so that the finiteness check below proves every
    element was written. Uninitialized device memory is usually finite, which would let a kernel that
    skipped a query tile or an accumulator group pass.
    """
    q, k, v = _inputs(1, 1024, 1024, 4, 4)
    out = torch.full(q.shape, float("nan"), dtype=torch.bfloat16, device=q.device)
    returned = mha_v4_sol_attn(q, k, v, BETA, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.isfinite(out.float()).all(), (
        f"{int(torch.isnan(out.float()).sum())} elements were left unwritten"
    )
    # And the buffer the kernel filled must hold what an allocated-for-you call returns.
    assert_launches_agree(out, mha_v4_sol_attn(q, k, v, BETA), "a caller-supplied out")


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
    """A supplied scale must reach the kernel and be the scale it actually applies.

    Passing the default explicitly has to be bitwise identical: `128 ** -0.5` and
    `1.0 / math.sqrt(128)` differ by one ULP as doubles, but the launcher narrows to float32 and both
    land on the same bits, so there is nothing for the kernel to disagree about.

    Then a different scale is checked against the oracle evaluated at that same scale, not merely for
    having moved the output. "The answer changed" is satisfied by a kernel that mangles the scale;
    "the answer matches the reference at that scale, to the FP8 floor measured at that scale" is not.
    """
    q, k, v = _inputs(1, 1024, 1024, 4, 4)
    default = mha_v4_sol_attn(q, k, v, BETA)
    explicit = mha_v4_sol_attn(q, k, v, BETA, softmax_scale=1.0 / math.sqrt(128))
    assert_launches_agree(explicit, default, "explicit default scale")

    half = 0.5 / math.sqrt(128)
    other = mha_v4_sol_attn(q, k, v, BETA, softmax_scale=half)
    expected, _ = _reference(q, k, v, softmax_scale=half)
    error = _rel_error(other, expected)
    floor = _fp8_arithmetic_floor(q, k, v, softmax_scale=half)
    assert error <= floor * FLOOR_MARGIN, (
        f"at half the default scale the kernel sits {error:.5f} from the reference against an FP8 "
        f"floor of {floor:.5f} (ratio {error / floor:.3f})"
    )
    # Halving the temperature flattens the distribution enough to move the output substantially;
    # measured 0.654. A kernel that quietly ignored the argument would land near zero here.
    assert _rel_error(other, default) > 0.1, (
        "the softmax scale did not reach the kernel"
    )


def test_repeated_calls_agree():
    """The sparse forward pass is reproducible, including across interleaved code objects.

    Interleaving a different kernel module and reallocating between launches is deliberate: an
    earlier revision of this code object lost bitwise reproducibility precisely when distinct modules
    were interleaved, and a lone back-to-back pair would not have caught it. Repeats are cheap here,
    so this walks the shape list rather than sampling one.
    """
    fp8 = native_fp8_format()
    for shape in SHAPES[:3]:
        q, k, v = _inputs(*shape)
        first = mha_v4_sol_attn(q, k, v, BETA)
        for repeat in range(4):
            churn = torch.empty(1 << 22, device="cuda", dtype=torch.bfloat16)
            if shape[3] == shape[4]:
                mha_v4(q, k, v, fp8, fp8, fp8)
            assert_launches_agree(
                mha_v4_sol_attn(q, k, v, BETA),
                first,
                f"repeat {repeat} of {shape} across an interleaved dense launch",
            )
            del churn


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


def _launch_sparse(
    q,
    k,
    v,
    mean_format=None,
    mean_view_dtype=None,
    sparse_mode=AttentionSparseMode.POOLED_CORRECTION,
):
    """Drive the private sparse launch directly, so dispatch-key rejections can be provoked.

    The public entrypoints cannot express an unsupported key, which is the point of them; reaching
    past them is the only way to assert that the launcher rejects one instead of quietly running the
    nearest row.
    """
    from aiter.ops.mha_v4 import _mha_v4_fwd_sparse_launch

    fp8 = native_fp8_format()
    q_quant, q_descale = quantize_fp8(q)
    k_quant, k_descale = quantize_fp8(k)
    v_quant, v_descale = quantize_fp8(v)
    routing = sol_attn_prepare(
        q_quant, k_quant, v_quant, BETA, SOL_ATTN_TS_QO, SOL_ATTN_TS_KV
    )
    mean_k, mean_v = routing["mean_k"], routing["mean_v"]
    if mean_view_dtype is not None:
        mean_k, mean_v = mean_k.view(mean_view_dtype), mean_v.view(mean_view_dtype)
    per_tensor = int(AttentionScaleMode.F32_PER_TENSOR)
    _mha_v4_fwd_sparse_launch(
        q_quant,
        k_quant,
        v_quant,
        q_descale,
        k_descale,
        v_descale,
        mean_k,
        mean_v,
        routing["kv_block_indices"],
        routing["lut_start"],
        routing["lut_count"],
        routing["block_bitmap"],
        torch.empty(q.shape, dtype=torch.bfloat16, device=q.device),
        int(fp8),
        int(fp8),
        int(fp8),
        per_tensor,
        per_tensor,
        per_tensor,
        int(fp8 if mean_format is None else mean_format),
        per_tensor,
        int(sparse_mode),
        128**-0.5,
    )


@pytest.mark.parametrize(
    "sparse_mode", [AttentionSparseMode.NONE, AttentionSparseMode.BLOCK_LUT]
)
def test_unimplemented_sparse_modes_are_refused(sparse_mode):
    """Only the pooled-correction row exists, and asking for another must fail rather than run one.

    A plain block LUT drops the skipped blocks instead of recovering them, so serving that request
    from this kernel would silently return a different function of the inputs. NONE is refused for the
    mirror-image reason: a dense request has no business arriving at the sparse launcher.
    """
    with pytest.raises(RuntimeError, match="pooled correction"):
        _launch_sparse(*_inputs(1, 1024, 1024, 4, 4), sparse_mode=sparse_mode)


def test_pooled_operand_format_selects_the_row():
    """mean_format is a real dispatch dimension, not decoration.

    The manifest row is keyed on the pooled operands' format and scale mode because pooling may only
    inherit K's descale under a per-tensor scale. Requesting pooled operands in a format no row
    provides therefore has to fail at manifest lookup; were the key ignored, this would launch the FP8
    row over pooled memory it had been told is int8.
    """
    with pytest.raises(RuntimeError, match="no MHA v4 kernel"):
        _launch_sparse(
            *_inputs(1, 1024, 1024, 4, 4),
            mean_format=AttentionFormat.INT8,
            mean_view_dtype=torch.int8,
        )


def test_manifest_keeps_dense_and_sparse_rows_disjoint():
    """No dense request can match the Sol-Attn row, by construction of the manifest key.

    The dense launcher looks up `mean_format` 0, `mean_scale_mode` 0 and sparse mode NONE, so this is
    a property of the manifest rather than of any output, and it is asserted there: every sparse row
    must carry a non-zero sparse mode and pooled-operand format, and no dense row may carry either.
    Comparing outputs cannot express it, because full density is the one setting where the two rows
    legitimately agree -- bitwise, as the test above requires.
    """
    manifest = (
        Path(aiter.__file__).resolve().parent.parent
        / "hsa"
        / arch_info.get_arch()
        / "fmha_v4_fwd"
        / "fmha_v4_fwd.csv"
    )
    rows = list(
        csv.DictReader(
            line
            for line in manifest.read_text().splitlines()
            if line and not line.startswith("#")
        )
    )
    sparse_rows = [row for row in rows if int(row["sparse"]) != 0]
    assert sparse_rows, f"no sparse row in {manifest.name}"
    for row in sparse_rows:
        assert int(row["mean_format"]) and int(row["mean_scale_mode"]), (
            f"sparse row {row['co_name']} declares no pooled-operand format, so a dense "
            "lookup would match it"
        )
    for row in (row for row in rows if int(row["sparse"]) == 0):
        assert not int(row["mean_format"]) and not int(row["mean_scale_mode"]), (
            f"dense row {row['co_name']} declares pooled operands it does not take"
        )
