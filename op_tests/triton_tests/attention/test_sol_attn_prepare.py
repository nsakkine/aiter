# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Sol-Attn routing/preprocessing: kernel ABI contract and torch.compile traceability.

sol_attn_prepare must be traceable under torch.compile(fullgraph=True). That is a hard requirement
of the entrypoint design, not a nice-to-have: a caller compiling an attention layer around Sol-Attn
must be able to trace straight through the routing, rather than hiding it in an opaque custom op of
its own. Two properties are what make it possible, and both are tested here rather than assumed:

  1. every output shape is a function of the input SHAPES alone, never of the values;
  2. no host-side branch reads device data.

Both were previously violated by the same code: a `counts[-1].item()` read of a token count the host
already knew from the shapes, and an `if empty.any()` guard around an idempotent term. This suite
pins the replacements as EQUIVALENT, not merely traceable.

The ASM kernel is not involved; this is pure Torch/Triton preprocessing and runs on any GPU.
"""

import pytest
import torch
import torch._dynamo

from aiter import dtypes
from aiter.ops.triton.attention.utils import (
    SOL_ATTN_TS_KV,
    SOL_ATTN_TS_QO,
    _sol_attn_pool_q,
    _sol_attn_route,
    sol_attn_prepare,
)

BETA = 0.5


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch._dynamo caches between tests so each gets a clean compile."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv, d=128, seed=0):
    """BSHD fp8 operands with a shared smooth positional component.

    Routing only has structure to find when attention concentrates; independent noise produces a
    near-uniform proxy, which would let a broken threshold still look plausible.
    """
    torch.manual_seed(seed)
    walk = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda") / (seqlen_k**0.5)
    traj = walk.cumsum(dim=1)
    traj = traj / traj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    amp = 2.0 * (d**0.5)

    q = torch.randn(batch, seqlen_q, nhead_q, d, device="cuda")
    k = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda")
    v = torch.randn(batch, seqlen_k, nhead_kv, d, device="cuda")
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
    q = q + amp * q_traj.repeat_interleave(nhead_q // nhead_kv, dim=2)
    k = k + amp * traj
    to_fp8 = lambda x: (x / x.abs().amax() * 448.0).to(dtypes.fp8)
    return to_fp8(q), to_fp8(k), to_fp8(v)


def _expected_shapes(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv, d=128):
    num_q_tiles = -(-seqlen_q // SOL_ATTN_TS_QO)
    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)
    work_items = batch * nhead_q * num_q_tiles
    bitmap_ds = 4 * (-(-num_kv_blocks // 128))
    return {
        "mean_k": ((batch, num_kv_blocks, nhead_kv, d), dtypes.fp8),
        "mean_v": ((batch, num_kv_blocks, nhead_kv, d), dtypes.fp8),
        "block_bitmap": ((work_items, bitmap_ds), torch.uint32),
        "kv_block_indices": ((work_items * num_kv_blocks,), torch.int32),
        "lut_start": ((work_items,), torch.int32),
        "lut_count": ((work_items,), torch.int32),
        "block_attn_mask": (
            (batch, nhead_q, num_q_tiles, num_kv_blocks),
            torch.bool,
        ),
    }


# Aligned and ragged sequence lengths, MHA and GQA. 9419 is a real Wan video shape and is a
# multiple of neither tile size.
SHAPES = [
    (1, 4096, 4096, 8, 8),
    (1, 4096, 4096, 8, 2),
    (2, 512, 1024, 4, 4),
    (1, 9419, 9419, 5, 5),
    (1, 257, 129, 4, 4),
]


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_output_contract(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """Shapes and dtypes the kernarg ABI depends on, at aligned and ragged sequence lengths."""
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_attn_prepare(q, k, v, BETA)

    for name, (shape, dtype) in _expected_shapes(
        batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
    ).items():
        assert tuple(prep[name].shape) == shape, f"{name} shape"
        assert prep[name].dtype == dtype, f"{name} dtype"
        assert prep[name].is_contiguous(), f"{name} must be contiguous"

    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)
    assert prep["num_kv_blocks"] == num_kv_blocks
    assert prep["num_q_tiles"] == -(-seqlen_q // SOL_ATTN_TS_QO)
    assert prep["bitmap_Ds"] == 4 * (-(-num_kv_blocks // 128))


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_shapes_do_not_depend_on_values(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """Same shapes for wildly different selections.

    This is the property that lets a caller trace through the routing. beta=-4 selects almost every
    block and beta=4 almost none, so any output whose size tracked the selection would differ.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    dense = sol_attn_prepare(q, k, v, -4.0)
    sparse = sol_attn_prepare(q, k, v, 4.0)

    assert dense["block_attn_mask"].sum() > sparse["block_attn_mask"].sum(), (
        "beta did not change the selection, so this test proves nothing"
    )
    for name in ("mean_k", "mean_v", "block_bitmap", "kv_block_indices", "lut_start"):
        assert dense[name].shape == sparse[name].shape, (
            f"{name} size is value-dependent"
        )


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_lut_and_bitmap_agree_with_mask(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv):
    """The two forms the kernel consumes must encode exactly the routed mask.

    The kernel reads selection twice, as a ragged LUT for the exact pass and as a bitmap for the
    approximate pass. A disagreement would double-count or drop a block's mass rather than fail.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_attn_prepare(q, k, v, BETA)
    mask = prep["block_attn_mask"]
    num_kv_blocks = mask.shape[-1]
    flat = mask.reshape(-1, num_kv_blocks)

    assert torch.equal(prep["lut_count"], flat.sum(-1, dtype=torch.int32))
    expected_start = (
        torch.cumsum(prep["lut_count"], 0, dtype=torch.int32) - prep["lut_count"]
    )
    assert torch.equal(prep["lut_start"], expected_start)

    # Every row, not a sample of them: nonzero() returns row-major order, so its column indices are
    # exactly the concatenation of the per-row block lists that lut_start and lut_count carve up.
    total = int(prep["lut_count"].sum())
    assert torch.equal(
        prep["kv_block_indices"][:total], flat.nonzero()[:, 1].to(torch.int32)
    ), "the LUT does not list the mask's selected blocks, in row-major order"

    bits = prep["block_bitmap"].to(torch.int64)
    unpacked = (
        (bits.unsqueeze(-1) >> torch.arange(32, device=bits.device))
        .bitwise_and(1)
        .bool()
        .reshape(bits.shape[0], -1)
    )
    assert torch.equal(unpacked[:, :num_kv_blocks], flat), "bitmap disagrees with mask"
    # Padding bits mean "already computed exactly" and are what clips the last tile's overhang.
    assert unpacked[:, num_kv_blocks:].all(), "bitmap padding bits must be set"


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_pooled_kv_matches_an_independent_block_mean(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """mean_k and mean_v must be the per-block mean of the stored values, tail block included.

    Nothing else pins their values. The routing tests consume mean_k rather than deriving it, and the
    kernel comparison feeds the same pooled tensors to both sides, so a pooling error would agree with
    itself everywhere and surface only as an unexplained accuracy gap. The ragged tail is the case
    that matters: dividing a short final block by the full block size scales it down by up to 128x
    while leaving every shape, dtype and contiguity check green.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_attn_prepare(q, k, v, BETA)
    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)

    for name, source in (("mean_k", k), ("mean_v", v)):
        pooled = prep[name]
        for block in range(num_kv_blocks):
            lo = block * SOL_ATTN_TS_KV
            hi = min(lo + SOL_ATTN_TS_KV, seqlen_k)
            expected = (source[:, lo:hi].float().sum(dim=1) / (hi - lo)).to(
                source.dtype
            )
            assert torch.equal(pooled[:, block], expected), (
                f"{name} block {block} covers tokens {lo}:{hi} ({hi - lo} of "
                f"{SOL_ATTN_TS_KV}) and does not equal their mean"
            )


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_every_work_item_selects_at_least_one_block(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """Kernel ABI invariant: the sparse prologue preloads LUT[0] unconditionally."""
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_attn_prepare(q, k, v, 8.0)  # threshold high enough to clear a row
    assert int(prep["lut_count"].min()) >= 1


def test_empty_row_rule_matches_the_guarded_form():
    """The unconditional empty-row term is equivalent to the `if empty.any()` version it replaced.

    Guarding it saved nothing but a cheap elementwise op, and cost a device sync and a graph break.
    A flat proxy is constructed here so that the rule actually fires: with an all-equal proxy row
    nothing exceeds mean + beta * std, so every row is empty and must fall back to its argmax.
    """
    batch, nhead, tiles, blocks, d = 1, 2, 3, 16, 128
    q_mean = torch.ones(batch, tiles, nhead, d, device="cuda")
    k_mean = torch.ones(batch, blocks, nhead, d, device="cuda")
    k_mean[:, 5] += 1e-3  # one block wins the argmax

    selected = _sol_attn_route(q_mean, k_mean, BETA, partial_tail=False)
    assert int(selected.sum(-1).min()) >= 1, "flat proxy must fall back to argmax"
    assert bool(selected[..., 5].all()), "fallback must keep the highest-proxy block"

    proxy = torch.einsum("bihd,bjhd->bhij", q_mean.float(), k_mean.float())
    guarded = proxy > (
        proxy.mean(-1, keepdim=True)
        + BETA * proxy.std(-1, unbiased=False, keepdim=True)
    )
    empty = ~guarded.any(dim=-1, keepdim=True)
    if bool(empty.any()):  # the host branch that used to be here
        guarded = guarded | (
            empty & torch.nn.functional.one_hot(proxy.argmax(-1), blocks).bool()
        )
    assert torch.equal(selected, guarded)


@pytest.mark.parametrize("seqlen_k", [128, 129, 255, 256, 1024, 9419])
def test_partial_tail_rule_matches_the_device_side_count(seqlen_k):
    """`seqlen_k % BLOCK_N != 0` is exactly the `counts[-1].item() != BLOCK_N` it replaced.

    The token-count tensor was only ever built from the shapes, so reading its last element back to
    the host round-tripped a value the host already had, for a full device sync.
    """
    num_kv_blocks = -(-seqlen_k // SOL_ATTN_TS_KV)
    pad = num_kv_blocks * SOL_ATTN_TS_KV - seqlen_k
    counts = torch.full(
        (num_kv_blocks,), SOL_ATTN_TS_KV, dtype=torch.int64, device="cuda"
    )
    if pad:
        counts[-1] = SOL_ATTN_TS_KV - pad
    assert (int(counts[-1].item()) != SOL_ATTN_TS_KV) == (
        seqlen_k % SOL_ATTN_TS_KV != 0
    )

    q, k, v = _operands(1, 512, seqlen_k, 4, 4)
    mask = sol_attn_prepare(q, k, v, 8.0)["block_attn_mask"]
    if seqlen_k % SOL_ATTN_TS_KV:
        assert bool(mask[..., -1].all()), (
            "a partial tail block must be computed exactly"
        )


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_routing_matches_reference_threshold(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """tau = mean_j(proxy) + beta * population_std_j(proxy), against an independent computation."""
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    prep = sol_attn_prepare(q, k, v, BETA)

    q_mean = _sol_attn_pool_q(q, SOL_ATTN_TS_QO)
    k_rep = prep["mean_k"].float().repeat_interleave(nhead_q // nhead_kv, dim=2)
    proxy = torch.einsum("bihd,bjhd->bhij", q_mean.float(), k_rep)
    tau = proxy.mean(-1, keepdim=True) + BETA * proxy.std(
        -1, unbiased=False, keepdim=True
    )
    expected = proxy > tau
    if seqlen_k % SOL_ATTN_TS_KV:
        expected[..., -1] = True
    empty = ~expected.any(dim=-1, keepdim=True)
    expected = expected | (
        empty & torch.nn.functional.one_hot(proxy.argmax(-1), proxy.shape[-1]).bool()
    )
    assert torch.equal(prep["block_attn_mask"], expected)


@pytest.mark.parametrize("batch, seqlen_q, seqlen_k, nhead_q, nhead_kv", SHAPES)
def test_compiles_fullgraph_and_matches_eager(
    batch, seqlen_q, seqlen_k, nhead_q, nhead_kv
):
    """fullgraph=True raises on any graph break, so compiling at all is the assertion.

    Equality with eager is checked too: a silently different routing would still be a valid mask and
    would only show up much later as an accuracy regression.
    """
    q, k, v = _operands(batch, seqlen_q, seqlen_k, nhead_q, nhead_kv)
    eager = sol_attn_prepare(q, k, v, BETA)
    compiled = torch.compile(sol_attn_prepare, fullgraph=True, dynamic=False)(
        q, k, v, BETA
    )

    for name, value in eager.items():
        if isinstance(value, torch.Tensor):
            if name == "kv_block_indices":
                # Only the spans named by lut_start/lut_count are meaningful; the rest of the
                # overallocated buffer is uninitialized in both paths.
                total = int(eager["lut_count"].sum())
                assert torch.equal(compiled[name][:total], value[:total]), name
            else:
                assert torch.equal(compiled[name], value), name
        else:
            assert compiled[name] == value, name


def test_no_graph_breaks_and_routing_is_in_the_graph():
    """Explicit break accounting, so a regression names the break instead of failing obscurely."""
    q, k, v = _operands(1, 4096, 4096, 8, 8)
    explained = torch._dynamo.explain(sol_attn_prepare)(q, k, v, BETA)
    assert explained.graph_break_count == 0, (
        f"graph breaks: {[str(r) for r in explained.break_reasons]}"
    )
    assert explained.graph_count == 1
