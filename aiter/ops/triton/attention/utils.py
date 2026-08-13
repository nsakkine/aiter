# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from aiter.ops.triton._triton_kernels.attention.block_lut import (
    block_attn_mask_to_lut_kernel,
)

# Tile geometry the gfx950 Sol-Attn ASM kernel is built for. SOL_ATTN_TS_KV in particular is the
# pooling block size whose log2 the kernel folds into its softmax bias as a constant, so pooling with
# any other value is silently wrong rather than an error.
SOL_ATTN_TS_QO = 256
SOL_ATTN_TS_KV = 128


def block_attn_mask_to_ragged_lut(
    block_attn_mask: torch.Tensor,
    num_heads: int | None = None,
    return_none_if_dense: bool = False,
    BLOCK_KB: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Convert a dense block attention mask to a ragged look-up table of KV block
    indices per (batch, head, q_block). Used for block-sparse attention with no
    per-iteration branching in the kernel.

    block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for
        same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks)
        for per-head masks. True = may attend, False = must not attend.
    num_heads: Required when block_attn_mask is 3D (number of Q heads). Ignored when 4D.
    return_none_if_dense: If True and the mask is all True (dense), return None so the
        caller can pass block_lut=None to fav3_sage_wrapper_func and use the dense path.
        Avoids building a very large LUT that can trigger munmap_chunk on MI300X/ROCm.
    Returns:
        kv_block_indices: 1D int32, concatenation of all KV block index lists.
        lut_start: 1D int32, length batch * num_heads * num_q_blocks. Index
            idx = batch_idx * (num_heads * num_q_blocks) + head_idx * num_q_blocks + q_block_idx.
        lut_count: 1D int32, same length as lut_start.
        When return_none_if_dense is True and the mask is all True, returns None instead.
    """
    device = block_attn_mask.device

    # 3D -> 4D: expand and fall through to 4D path
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        batch, num_q_blocks, num_kv_blocks = block_attn_mask.shape
        if return_none_if_dense and block_attn_mask.all():
            return None
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(
            batch, num_heads, num_q_blocks, num_kv_blocks
        )

    # 4D: (batch, num_heads, num_q_blocks, num_kv_blocks) — GPU vectorized path
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    if return_none_if_dense and block_attn_mask.all():
        return None

    counts = block_attn_mask.sum(dim=-1, dtype=torch.int32)
    lut_count = counts.reshape(-1)
    lut_start = torch.cumsum(lut_count, dim=0, dtype=torch.int32) - lut_count

    # NOTE: Overallocating the LUT is a waste of memory, but the
    # alternative lut_count.sum(), will cause graph break with torch compile.
    max_count = batch * num_heads * num_q_blocks * num_kv_blocks
    kv_block_indices = torch.empty(max_count, dtype=torch.int32, device=device)
    block_attn_mask_to_lut_kernel(
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        BLOCK_KB=BLOCK_KB,
    )

    return kv_block_indices, lut_start, lut_count


def _sol_attn_pool_kv_quant(
    k_quant: torch.Tensor, v_quant: torch.Tensor, BLOCK_N: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool stored (already quantized) K/V into one row per KV block, in the SOURCE dtype.

    The block mean is accumulated in fp32 over the raw stored values and rounded back to the
    source dtype, so K's and V's PER-TENSOR descales stay valid for the pooled tensors
    (mean(x) * descale == mean(x * descale)). The kernel folds those descales into the softmax
    temperature and the epilogue, so a separate scale on either mean tensor would corrupt the
    exact/approximate mix. Rounding a mean of in-range values back to fp8 cannot overflow.

    This identity is what ties pooling to per-tensor scaling: a block-granular scale image
    (E8M0 per 1x32) spans several scale blocks per pooled row and has no single descale to
    inherit, so a non-per-tensor variant needs its own pooled-operand scale contract rather than
    this function.

    Returns (mean_k, mean_v), BSHD (batch, num_kv_blocks, nheads_kv, d) and contiguous. The
    allocation stops at num_kv_blocks: the approximate pass processes whole 128-block groups, but
    it masks the overhang columns before it loads their pooled rows, so it never reads past the end.
    """
    batch, seqlen_k, nhead_kv, _ = k_quant.shape
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    pad = num_kv_blocks * BLOCK_N - seqlen_k
    counts = torch.full(
        (num_kv_blocks,), BLOCK_N, dtype=torch.float32, device=k_quant.device
    )
    if pad:
        counts[-1] = BLOCK_N - pad
    denom = counts.view(1, num_kv_blocks, 1, 1)

    def _pool(x):
        xf = x.float()
        if pad:
            xf = F.pad(xf, (0, 0, 0, 0, 0, pad))
        xf = xf.reshape(batch, num_kv_blocks, BLOCK_N, nhead_kv, x.shape[3])
        return (xf.sum(dim=2) / denom).to(x.dtype).contiguous()

    return _pool(k_quant), _pool(v_quant)


def _sol_attn_pool_q(q: torch.Tensor, BLOCK_M: int) -> torch.Tensor:
    """
    Pool Q into one representative row per query tile (the paper's Q-bar), fp32.

    q: (batch, seqlen_q, nheads_q, d) -> (batch, num_q_tiles, nheads_q, d)
    """
    batch, seqlen_q, nhead_q, d = q.shape
    num_q_tiles = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    pad = num_q_tiles * BLOCK_M - seqlen_q
    counts = torch.full((num_q_tiles,), BLOCK_M, dtype=torch.float32, device=q.device)
    if pad:
        counts[-1] = BLOCK_M - pad
    qf = q.float()
    if pad:
        qf = F.pad(qf, (0, 0, 0, 0, 0, pad))
    qf = qf.reshape(batch, num_q_tiles, BLOCK_M, nhead_q, d)
    return qf.sum(dim=2) / counts.view(1, num_q_tiles, 1, 1)


def _sol_attn_route(
    q_mean: torch.Tensor,
    k_mean: torch.Tensor,
    beta: float,
    partial_tail: bool,
) -> torch.Tensor:
    """
    Query-dependent threshold routing, paper Eq. (3)-(5) and (7).

        proxy[b, h, i, j] = q_mean[b, i, h, :] . k_mean[b, j, h, :]
        tau[b, h, i]      = mean_j(proxy) + beta * population_std_j(proxy)
        selected          = proxy > tau

    The std is the population std, which is what the Eq. (5) closed form over the pooled-key first
    and second moments computes. Selection is invariant to a positive rescale of the logits (mu and
    sigma scale with proxy), so the softmax scale and the per-tensor descales are deliberately NOT
    applied here: host and kernel cannot disagree about routing because of a scale factor.

    That invariance is exact in real arithmetic but not in fp32: routing on quantized values here
    and on dequantized values in a reference rounds the last bit differently, so a block whose proxy
    sits within ~1e-7 * sigma of tau can land on either side (measured: 1 block in 4096 for seqlen
    4096, GQA 4). Both answers are equally valid, but it means the mask is the HOST's to own:
    compare a kernel run against a reference evaluated on the mask this function returned, never
    against a mask the reference rerouted itself.

    partial_tail: whether the last KV block holds fewer than BLOCK_N real tokens. This is a
    property of the shapes (seqlen_k % BLOCK_N != 0), not of the values, and is passed in rather
    than recovered from a token-count tensor so that routing stays free of host-side reads of
    device data and therefore traceable under torch.compile.

    Returns (batch, nheads_q, num_q_tiles, num_kv_blocks) bool, True == compute exactly.
    """
    g = q_mean.shape[2] // k_mean.shape[2]
    k_rep = k_mean.float().repeat_interleave(g, dim=2)
    proxy = torch.einsum("bihd,bjhd->bhij", q_mean.float(), k_rep)
    mu = proxy.mean(dim=-1, keepdim=True)
    sigma = proxy.std(dim=-1, unbiased=False, keepdim=True)
    selected = proxy > (mu + beta * sigma)
    # A partial tail block reaching the approximate branch would be scaled by the kernel's constant
    # block-size factor, which is only exact for a full block.
    if partial_tail:
        selected[..., -1] = True
    # lut_count >= 1 for every work item, per the kernel ABI: the sparse prologue preloads LUT[0]
    # unconditionally, and the approximate pass runs a seeded softmax that cannot establish a
    # running max by itself. A nearly flat proxy row can otherwise clear the whole row, so keep its
    # highest-proxy block. Applied unconditionally: the term is empty for rows that already selected
    # something, so guarding it on empty.any() would only trade a cheap elementwise op for a device
    # sync and a graph break.
    empty = ~selected.any(dim=-1, keepdim=True)
    return selected | (empty & F.one_hot(proxy.argmax(dim=-1), proxy.shape[-1]).bool())


def sol_attn_prepare(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    v_quant: torch.Tensor,
    beta: float,
    BLOCK_M: int = SOL_ATTN_TS_QO,
    BLOCK_N: int = SOL_ATTN_TS_KV,
    num_heads: int | None = None,
) -> dict[str, Any]:
    """
    Build every host-side input of the gfx950 Sol-Attn kernel (arXiv 2607.24027) from Q and the
    quantized K/V: the pooled K/V of the approximate branch and the routed block selection in both
    of the forms the kernel consumes (ragged LUT + bitmap).

    Every output shape here is a function of the INPUT SHAPES alone, and no host-side branch reads
    device data, so this whole function is traceable under torch.compile(fullgraph=True). That is a
    hard requirement, not an accident: it is what lets a caller compile an attention layer around
    Sol-Attn without wrapping the routing in its own opaque custom op.

    q: (batch, seqlen_q, nheads_q, d), any dtype; only used for routing, which is scale invariant,
        so the stored fp8 Q can be passed directly.
    k_quant: (batch, seqlen_k, nheads_kv, d) already quantized, typically fp8 e4m3.
    v_quant: (batch, seqlen_k, nheads_kv, d_v) already quantized.
    beta: routing threshold, tau = mean_j(proxy) + beta * population_std_j(proxy).
    num_heads: optional cross-check on nheads_q.

    Returns a dict with:
        mean_k, mean_v: pooled K/V in K's / V's own dtype, BSHD
            (batch, num_kv_blocks, nheads_kv, d) and contiguous, reusing the SOURCE descales.
        block_bitmap: uint32 (num_work_items, bitmap_Ds), contiguous, bit j of word j // 32 set
            == KV block j selected for that work item. bitmap_Ds is 4 * ceil(num_kv_blocks / 128)
            and the bits above num_kv_blocks are SET; see the packing comment below.
        kv_block_indices, lut_start, lut_count: the ragged LUT, int32. kv_block_indices is
            overallocated by block_attn_mask_to_ragged_lut; only the spans are meaningful.
        num_kv_blocks, bitmap_Ds, num_q_tiles: kernarg scalars / grid geometry.
        block_attn_mask: the routed mask, for reference comparisons.

    block_bitmap, lut_start and lut_count are all indexed by the kernel's
        lut_idx = (b * nheads_q + h) * num_q_tiles + q_tile
    and are laid out contiguously in that order, so the bitmap and the LUT cannot disagree: both are
    derived from one boolean mask.
    """
    if q.dim() != 4 or k_quant.dim() != 4 or v_quant.dim() != 4:
        raise ValueError("q, k_quant and v_quant must be 4D (batch, seqlen, nheads, d)")
    batch, seqlen_q, nhead_q, _ = q.shape
    seqlen_k, nhead_kv = k_quant.shape[1], k_quant.shape[2]
    if k_quant.shape[0] != batch or v_quant.shape[:3] != k_quant.shape[:3]:
        raise ValueError("k_quant and v_quant must share (batch, seqlen_k, nheads_kv)")
    if nhead_q % nhead_kv != 0:
        raise ValueError("nheads_q must be a multiple of nheads_kv")
    if num_heads is not None and num_heads != nhead_q:
        raise ValueError(f"num_heads {num_heads} does not match q's {nhead_q}")

    num_q_tiles = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    mean_k, mean_v = _sol_attn_pool_kv_quant(k_quant, v_quant, BLOCK_N)
    # Route on the pooled values the kernel will actually load, i.e. after the fp8 rounding.
    block_attn_mask = _sol_attn_route(
        _sol_attn_pool_q(q, BLOCK_M),
        mean_k,
        beta,
        partial_tail=seqlen_k % BLOCK_N != 0,
    )

    kv_block_indices, lut_start, lut_count = block_attn_mask_to_ragged_lut(
        block_attn_mask
    )
    lut_start = lut_start.to(torch.int32)
    lut_count = lut_count.to(torch.int32)

    # Bitmap: same mask, packed 32 blocks per uint32 word, num_work_items rows of bitmap_Ds words in
    # lut_idx order.
    #
    # The row length is rounded up to whole 128-block groups (4 words), not to a single word: the
    # kernel reads one group per approximate tile with a single s_load_dwordx4 at byte offset
    # 16 * tile, so a row that is not a multiple of 4 words misaligns every tile after the first.
    # The padding bits are SET, because a set bit means "already computed exactly" and so masks that
    # column out of the approximate pass; that is what clips the last tile's overhang, and it is why
    # the kernel needs no masked-tail path at all. Clearing them instead would let the pooled rows
    # past num_kv_blocks contribute spurious mass.
    num_work_items = batch * nhead_q * num_q_tiles
    bitmap_Ds = 4 * ((num_kv_blocks + 127) // 128)
    bits = block_attn_mask.reshape(num_work_items, num_kv_blocks)
    if bitmap_Ds * 32 != num_kv_blocks:
        bits = F.pad(bits, (0, bitmap_Ds * 32 - num_kv_blocks), value=True)
    weights = (1 << torch.arange(32, device=q.device, dtype=torch.int64)).view(1, 1, 32)
    block_bitmap = (
        (bits.reshape(num_work_items, bitmap_Ds, 32).to(torch.int64) * weights)
        .sum(dim=-1)
        .to(torch.uint32)
        .contiguous()
    )

    return {
        "mean_k": mean_k,
        "mean_v": mean_v,
        "block_bitmap": block_bitmap,
        "kv_block_indices": kv_block_indices,
        "lut_start": lut_start,
        "lut_count": lut_count,
        "num_kv_blocks": num_kv_blocks,
        "bitmap_Ds": bitmap_Ds,
        "num_q_tiles": num_q_tiles,
        "block_attn_mask": block_attn_mask,
    }
