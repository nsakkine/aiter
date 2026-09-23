# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sol-Attn vs keep-or-drop sparse over the SAME preset selection, swept by sparsity.

Routing makes a Sol-Attn run's density a property of the data and beta, so timing a
routed run against a dense one mostly measures whichever density beta happened to pick.
Pinning the selection instead isolates what actually differs between mode 1 and mode 2:
the approximate pass over the pooled K/V. Both kernels here are handed the identical
mask.

Two caveats on reading the delta. The kernels are separately compiled ASM objects, so at
matched density they can differ by more than the extra work -- a mode-2 row measuring
FASTER than its mode-1 counterpart means the two binaries schedule the same exact pass
differently, not that the sweep is free. And the cleanest number in the table is the
sparsity 1.0 row, where no block is exact and the difference is the sweep alone.

The mask is random rather than routed on purpose. Density is the only property of it
that the kernels' work depends on -- neither has data-dependent control flow, and the
pooled VALUES cannot change the instruction count -- so a random mask at a density times
like a routed one at that density while being something we can dial. Accuracy is
deliberately not reported: a random selection has nothing to do with the one routing
would choose, and the accuracy of the real thing is what op_tests/test_mha_v4.py pins.

Shape is Wan 2.2 A14B self-attention (dim 5120, 40 heads, head_dim 128, patch (1,2,2),
VAE stride (4,8,8)), which is MHA rather than GQA. Ulysses shards heads inside
attention and the sequence outside it, so degree U keeps the full token count and leaves
40/U heads on each device.

Usage:
    python bench_sol_attn_vs_sparse.py                      # 720p, Ulysses 1 and 8
    python bench_sol_attn_vs_sparse.py --resolution 480p --ulysses 8
"""

import argparse
import sys

import torch
import triton

from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4_kv_tile,
    mha_v4_packed,
    mha_v4_q_multiplier,
    native_fp8_format,
    quantize_fp8,
    quantize_fp8_rotated,
    quantize_mxfp8_k,
    quantize_mxfp8_q,
    scale_modes_for_formats,
)
from aiter.ops.triton.attention.utils import (
    SOL_ATTN_TS_QO,
    block_attn_mask_to_ragged_lut,
    sol_attn_prepare,
)

# Wan 2.2 A14B transformer.
WAN_HEADS = 40
WAN_HEAD_DIM = 128
WAN_PATCH = (1, 2, 2)
WAN_VAE_STRIDE = (4, 8, 8)

# (width, height, frames) for the two sampling resolutions Wan 2.2 ships.
RESOLUTIONS = {"720p": (1280, 720, 81), "480p": (832, 480, 81)}


def wan_self_attn_tokens(resolution: str) -> int:
    """Video tokens entering self-attention: VAE downsample, then 3D patchify.

    Self-attention in Wan runs over the video tokens alone; the text conditioning
    arrives through the separate cross-attention, so its 512 tokens do not belong in
    this sequence length.
    """
    width, height, frames = RESOLUTIONS[resolution]
    stride_t, stride_h, stride_w = WAN_VAE_STRIDE
    patch_t, patch_h, patch_w = WAN_PATCH
    latent_t = (frames - 1) // stride_t + 1
    return (
        (latent_t // patch_t)
        * (height // stride_h // patch_h)
        * (width // stride_w // patch_w)
    )


def build_operands(seqlen: int, heads: int, recipe: str, device="cuda"):
    """Quantize one set of BF16 operands through the production path for this recipe."""
    torch.manual_seed(0)
    shape = (1, seqlen, heads, WAN_HEAD_DIM)
    q = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(*shape, device=device, dtype=torch.bfloat16)

    fp8 = native_fp8_format()
    softmax_scale = WAN_HEAD_DIM**-0.5

    if recipe == "fp8":
        q_quant, q_descale = quantize_fp8_rotated(q)
        k_quant, k_descale = quantize_fp8_rotated(k)
        v_quant, v_descale = quantize_fp8(v)
        formats = (fp8, fp8, fp8)
        scale_modes = scale_modes_for_formats(*formats)
        # Per-tensor descales survive pooling untouched, so no pooled scale is passed.
        prepare_kwargs = {}
    elif recipe == "mxfp8":
        q_quant, q_descale = quantize_mxfp8_q(q, mha_v4_q_multiplier(softmax_scale))
        k_quant, k_descale = quantize_mxfp8_k(k)
        v_quant, v_descale = quantize_fp8(v)
        formats = (fp8, fp8, fp8)
        scale_modes = (
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.F32_PER_TENSOR,
        )
        # K is block granular, so its pooled image needs a scale of its own; V is per-
        # tensor.
        prepare_kwargs = {"k_scale": k_descale}
    elif recipe in ("bf16", "bf16fp8"):
        # Q and K stay BF16, so they have no descale; the placeholder keeps the launch signature
        # uniform and the NONE scale mode makes the kernel ignore it. Only V can be quantized,
        # and only to per-tensor FP8, which survives pooling as the FP8 recipe's does.
        q_quant, q_descale = q, q
        k_quant, k_descale = k, k
        v_is_fp8 = recipe == "bf16fp8"
        v_quant, v_descale = quantize_fp8(v) if v_is_fp8 else (v, v)
        formats = (
            AttentionFormat.BF16,
            AttentionFormat.BF16,
            fp8 if v_is_fp8 else AttentionFormat.BF16,
        )
        scale_modes = scale_modes_for_formats(*formats)
        prepare_kwargs = {}
    else:
        raise ValueError(f"unknown recipe {recipe!r}")

    del q, k, v
    torch.cuda.empty_cache()
    return dict(
        tensors=(q_quant, k_quant, v_quant, q_descale, k_descale, v_descale),
        formats=formats,
        scale_modes=scale_modes,
        prepare_kwargs=prepare_kwargs,
        softmax_scale=softmax_scale,
    )


def time_pair(operands, mask, warmup, rep):
    """Time the sparse and Sol-Attn launches over one selection.

    sol_attn_prepare owns the mask it returns -- it forces a short tail block onto the
    exact pass, which the approximate branch's constant full-block factor cannot
    represent -- so the sparse LUT is built from THAT mask, not the one handed in.
    Otherwise the two kernels would be timed over selections differing by one block.
    """
    tensors = operands["tensors"]
    q_quant, k_quant, v_quant, _, k_descale, _ = tensors
    launch_args = (*tensors, *operands["formats"], *operands["scale_modes"])
    softmax_scale = operands["softmax_scale"]

    plan = sol_attn_prepare(
        q_quant,
        k_quant,
        v_quant,
        # The manifest's KV tile, not the default: gfx942 re-tiles to 64 where gfx950 uses
        # 128, and the mask this is handed was built at the manifest's size. Leaving it
        # defaulted made the two disagree by a factor of two and the call fail outright,
        # which is why this bench only ever ran on gfx950.
        BLOCK_N=mha_v4_kv_tile(),
        block_attn_mask=mask,
        **operands["prepare_kwargs"],
    )
    used = plan["block_attn_mask"]
    kv_idx, lut_start, lut_count = block_attn_mask_to_ragged_lut(used)
    lut_start, lut_count = lut_start.to(torch.int32), lut_count.to(torch.int32)

    def sparse():
        return mha_v4_packed(
            *launch_args,
            softmax_scale=softmax_scale,
            kv_block_indices=kv_idx,
            lut_start=lut_start,
            lut_count=lut_count,
        )

    def sol_attn():
        return mha_v4_packed(
            *launch_args,
            softmax_scale=softmax_scale,
            kv_block_indices=plan["kv_block_indices"],
            lut_start=plan["lut_start"],
            lut_count=plan["lut_count"],
            mean_k=plan["mean_k"],
            mean_v=plan["mean_v"],
            block_bitmap=plan["block_bitmap"],
            mean_k_scale=plan["mean_k_scale"],
            mean_v_scale=plan["mean_v_scale"],
        )

    density = used.float().mean().item()
    sparse_ms = triton.testing.do_bench(sparse, warmup=warmup, rep=rep)
    sol_ms = triton.testing.do_bench(sol_attn, warmup=warmup, rep=rep)
    del plan, kv_idx, lut_start, lut_count
    torch.cuda.empty_cache()
    return density, sparse_ms, sol_ms


def run_degree(degree, tokens, sparsities, recipe, warmup, rep):
    heads = WAN_HEADS // degree
    # Mode 1 requires a key length that is a whole number of KV tiles, and Wan's token
    # counts are
    # not (75600 leaves 80, 32760 leaves 120). Padding up is what a deployment does
    # regardless, and
    # it also keeps the two kernels honest: a short tail block is forced onto Sol-Attn's
    # exact pass
    # and would otherwise be one block of selection the sparse row never saw.
    block_n = mha_v4_kv_tile()
    seqlen = -(-tokens // block_n) * block_n
    num_q_tiles = -(-seqlen // SOL_ATTN_TS_QO)
    num_kv_blocks = seqlen // block_n
    dense_flops = 2.0 * heads * seqlen * seqlen * 2 * WAN_HEAD_DIM

    print(
        f"\n=== Wan 2.2 self-attn, Ulysses {degree}: "
        f"{heads} heads x {seqlen} tokens x d{WAN_HEAD_DIM}, {recipe} ===\n"
        f"{tokens} real tokens padded to {seqlen} ({num_q_tiles} query tiles x "
        f"{num_kv_blocks} KV blocks), {dense_flops / 1e12:.1f} TFLOP dense\n"
    )
    print(
        f"{'sparsity':>9} {'density':>8} {'sparse(ms)':>11} {'sol-attn(ms)':>13}"
        f" {'delta':>8} {'sol vs dense':>13}"
    )
    print("-" * 70)

    operands = build_operands(seqlen, heads, recipe)
    dense_ms = _time_dense(operands, warmup, rep)
    rows = []
    for sparsity in sparsities:
        torch.manual_seed(int(sparsity * 1000))
        mask = (
            torch.rand(1, heads, num_q_tiles, num_kv_blocks, device="cuda") > sparsity
        )
        density, sparse_ms, sol_ms = time_pair(operands, mask, warmup, rep)
        rows.append((sparsity, density, sparse_ms, sol_ms))
        print(
            f"{sparsity:>9.1f} {100 * density:>7.1f}% {sparse_ms:>11.4f}"
            f" {sol_ms:>13.4f} {(sol_ms / sparse_ms - 1) * 100:>+7.1f}%"
            f" {dense_ms / sol_ms:>12.2f}x"
        )
        del mask
        torch.cuda.empty_cache()

    print(f"\ndense (no mask, mode 0): {dense_ms:.4f} ms")
    del operands
    torch.cuda.empty_cache()
    return rows


def _time_dense(operands, warmup, rep):
    launch_args = (*operands["tensors"], *operands["formats"], *operands["scale_modes"])

    def dense():
        return mha_v4_packed(
            *launch_args, softmax_scale=operands["softmax_scale"]
        )

    return triton.testing.do_bench(dense, warmup=warmup, rep=rep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        default="720p",
        choices=sorted(RESOLUTIONS),
        help="Wan sampling size",
    )
    parser.add_argument(
        "--ulysses",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Ulysses degrees; each leaves 40/degree heads on a device",
    )
    parser.add_argument(
        "--recipe", default="fp8", choices=["fp8", "mxfp8", "bf16", "bf16fp8"]
    )
    parser.add_argument("--warmup", type=int, default=25, help="do_bench warmup ms")
    parser.add_argument("--rep", type=int, default=100, help="do_bench rep ms")
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[round(0.1 * i, 1) for i in range(1, 11)],
        help="block sparsity levels; exact density is 1 - sparsity",
    )
    args = parser.parse_args()

    tokens = wan_self_attn_tokens(args.resolution)
    for degree in args.ulysses:
        if WAN_HEADS % degree:
            raise ValueError(
                f"Ulysses degree {degree} does not divide {WAN_HEADS} heads"
            )
        run_degree(degree, tokens, args.sparsity, args.recipe, args.warmup, args.rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
