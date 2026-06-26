#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Block-sparse FMHA forward (Sage i8fp8, hd=128, gfx950).
// Sibling of mha_fwd.h / fmha_fwd_v3_args, extended with the 3 LUT pointers
// that the hand-written ASM kernel consumes at kernarg offsets 0x290/0x2A0/
// 0x2B0 (see /workspace/mi350_fmha_hd128_i8fp8_sparse.py).

#include "aiter_hip_common.h"
#include "mha_fwd.h"

namespace aiter {

// Block-sparse args. Augments the dense mha_fwd_args (no inheritance to keep
// the existing struct's POD-ness intact) with the 3 LUT tensor pointers
// produced by aiter.ops.triton.attention.utils.block_attn_mask_to_ragged_lut.
struct mha_fwd_sparse_args : public mha_fwd_args
{
    const void* kv_block_indices_ptr; // int32, shape [lut_count.sum()]
    const void* lut_start_ptr;        // int32, shape [B*HQ*num_q_blocks]
    const void* lut_count_ptr;        // int32, same shape
    // VSA (Vector-relieved Sparse Attention): per-(b,head,q_block) number of
    // leading (highest-priority) KV blocks to process with a live, max-updating
    // softmax before freezing m. int32, same shape as lut_count. nullptr =>
    // freezing disabled (kernel falls back to n_freeze = n_blocks, i.e. plain
    // block-sparse behaviour, bit-for-bit).
    const void* lut_freeze_ptr;

    // ---- Persistent kernel (grid-stride over Q-tiles); only consumed by the
    // fmha_fwd_v3_fp8_sparse_persistent dispatcher + the _PERSISTENT=True .co. ----
    // work_table: int32[total_tiles], entry k = packed q | (h<<16) | (b<<24) for
    // the k-th tile to process (typically LPT-sorted by lut_count descending).
    // nullptr on the non-persistent path.
    const void* work_table_ptr = nullptr;
    uint32_t    num_wgs        = 0;   // persistent grid size (== gridDim.x). 0 => dispatcher picks.
    uint32_t    total_tiles    = 0;   // = batch * nhead_q * num_q_blocks (== work_table.numel()).
};

// On-device kernarg blob: same 656 bytes as fmha_fwd_v3_args + 64 bytes
// (4 ptr slots with the same p2 padding convention).
struct __attribute__((packed)) fmha_fwd_v3_sparse_args : public fmha_fwd_v3_args
{
    const void* ptr_kv_block_indices;
    p2 _ppad_kv;
    const void* ptr_lut_start;
    p2 _ppad_ls;
    const void* ptr_lut_count;
    p2 _ppad_lc;
    const void* ptr_lut_freeze; // VSA: kernarg offset 0x2C0
    p2 _ppad_lf;
};

static_assert(sizeof(fmha_fwd_v3_sparse_args) == 720,
              "fmha_fwd_v3_sparse_args must be exactly 720 bytes "
              "(matches the @kernel(_kernarg_raw size=720) in "
              "mi350_fmha_hd128_fp8_sparse.py: 656 dense + 4*16 LUT ptr slots).");

// Persistent-kernel kernarg blob: the 720-byte sparse layout + 3 trailing fields
// at offsets 0x2D0/0x2E0/0x2E4, padded to 752 to match the @kernel(_kernarg_raw
// size=752) emitted when mi350_fmha_hd128_fp8_sparse.py is built with
// _PERSISTENT=True. The non-persistent .co never sees these bytes.
struct __attribute__((packed)) fmha_fwd_v3_sparse_persistent_args
    : public fmha_fwd_v3_sparse_args
{
    const void* ptr_work_table; // kernarg offset 0x2D0 (int32[total_tiles])
    p2 _ppad_wt;
    uint32_t s_num_wgs;         // 0x2E0 grid-stride
    uint32_t s_total_tiles;     // 0x2E4 loop bound
    uint64_t _tail_pad;         // pad 744 -> 752 (matches the .co kernarg segment)
};

static_assert(sizeof(fmha_fwd_v3_sparse_persistent_args) == 752,
              "fmha_fwd_v3_sparse_persistent_args must be exactly 752 bytes "
              "(matches @kernel(_kernarg_raw size=752) when _PERSISTENT=True: "
              "720 sparse + work_table(16) + num_wgs(4) + total_tiles(4) + pad(8)).");

// Sparse dispatcher. Returns the launch time in ms, -1 on unsupported config.
float fmha_fwd_v3_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// Sparse mxfp4 sibling. Same kernarg layout, different .co
// (fwd_hd128_mxfp4_sparse.co) generated from
// /workspace/mi350_fmha_hd128_mxfp4_sparse.py. Q/K are fp4-packed
// (caller tensor dtype int8/uint8 with last dim = head_dim/2 = 64
// for hd=128), V is fp8, Q/K scales are E8M0 per-block uint8 bytes
// and V descale is fp32 per output channel -- but on the kernel side
// none of those buffer dtypes are baked into the kernarg, so the
// dispatcher just forwards base pointers and lets the wrapper
// validate shapes (see asm_mha_fwd_sparse.cu::fmha_v3_fwd_mxfp4_sparse).
float fmha_fwd_v3_mxfp4_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// Sorted-dispatch mxfp4 sparse sibling. Same data + 720-byte base contract plus
// a.work_table_ptr / a.total_tiles. One WG per tile on a flat grid
// gridDim=(total_tiles,1,1); each WG reads work_table[wg_id] (LPT, heavy tiles first) and decodes
// its (q,h,b). Routes to fwd_hd128_mxfp4_sparse_sorted.co (752-byte kernarg). There is NO persistent
// sub-mode for mxfp4 (SGPR budget), so unlike the fp8 path this is sorted-only.
float fmha_fwd_v3_mxfp4_sparse_sorted(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// DENSE mxfp4 sibling (no LUT / no block sparsity): processes every KV tile.
// Reuses the same 656-byte fmha_fwd_v3_args prefix (init_sparse_v3_args) and the
// same mxfp4 numeric contract (fp4-packed Q/K + E8M0 per-block scales via MFMA,
// fp8 V, bf16 out); only the symbol + .co (fwd_hd128_mxfp4.co) differ. Grid +
// bdx match the sparse path. See asm_mha_fwd_sparse.cu::fmha_v3_fwd_mxfp4.
float fmha_fwd_v3_mxfp4(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// Sparse fp8 sibling. Same kernarg layout, different .co
// (fwd_hd128_fp8_sparse.co) generated from
// /workspace/mi350_fmha_hd128_fp8_sparse.py. Q/K/V are all fp8 (E4M3)
// and the descales are per-tensor fp32, so the dispatcher reuses the
// i8fp8 init_sparse_v3_args path verbatim (in_bpe=1 for fp8).
//
// q128kv64=true routes to the finer-KV-granularity fork
// (fwd_hd128_fp8_sparse_q128kv64.co, kTileKV=64). Stage 1 shares the same
// kernarg/grid/bdx, so only the symbol + .co change; the caller MUST build the
// LUT with BLOCK_N=64 to match the kernel's 64-row KV stride.
float fmha_fwd_v3_fp8_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s,
                             bool q128kv64 = false);

// Work-table variant of the fp8 sparse path. Same data contract plus
// a.work_table_ptr / a.total_tiles. Two sub-modes:
//   * sorted_dispatch=true (recommended): one WG per tile, flat grid
//     gridDim=(total_tiles,1,1); each WG reads work_table[wg_id]. Keeps the
//     hardware scheduler work-conserving and only fixes dispatch ORDER
//     (LPT, heavy tiles first). Routes to fwd_hd128_fp8_sparse_sorted.co.
//   * sorted_dispatch=false: persistent grid-stride; fixed 1-D grid of a.num_wgs
//     workgroups (auto-sized to occupancy when 0). Routes to
//     fwd_hd128_fp8_sparse_persistent.co.
float fmha_fwd_v3_fp8_sparse_persistent(mha_fwd_sparse_args a,
                                        const ck_tile::stream_config& s,
                                        bool sorted_dispatch = false);

} // namespace aiter
