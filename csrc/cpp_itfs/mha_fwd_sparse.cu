// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Block-sparse FMHA forward (Sage i8fp8, hd=128, gfx950) C++ dispatcher.
//
// Mirrors aiter::fmha_fwd_v3 in mha_fwd.cu but routes to the hand-written
// sparse ASM kernel (fwd_hd128_i8fp8_sparse.co). The 720-byte kernarg blob is
// the dense 656-byte fmha_fwd_v3_args layout plus 4 trailing pointers
// (kv_block_indices, lut_start, lut_count, lut_freeze) -- see
// /workspace/mi350_fmha_hd128_i8fp8_sparse.py docstring. lut_freeze (VSA) is
// only consumed by the fp8 .co today; the i8fp8/mxfp4 .co ignore the slot.

#include "mha_fwd_sparse.h"
#include "aiter_hip_common.h"
#include <memory>
#include <string>

namespace aiter {

// Hardcoded for the single shape this kernel currently supports.
// (BLOCK_M, BLOCK_N) = (256, 128); hd_q = hd_v = 128; non-causal.
// The mxfp4 sibling shares the SAME 720-byte kernarg layout (sparse args
// are only the trailing 48-byte LUT pointer block, identical for both).
static constexpr int      kSparseTileQ = 256;
static constexpr int      kSparseTileN = 128;
static constexpr int      kSparseBdx   = 512;
static constexpr const char* kSparseKernelName =
    "_ZN5aiter35fmha_fwd_hd128_i8fp8_sparse_gfx950E";
static constexpr const char* kSparseCoName =
    "fmha_v3_fwd/fwd_hd128_i8fp8_sparse.co";
static constexpr const char* kSparseMxfp4KernelName =
    "_ZN5aiter35fmha_fwd_hd128_mxfp4_sparse_gfx950E";
static constexpr const char* kSparseMxfp4CoName =
    "fmha_v3_fwd/fwd_hd128_mxfp4_sparse.co";
// Sorted-dispatch mxfp4 sparse sibling: one WG per tile, flat grid indexed through the LPT work
// table. SAME kernel symbol as the default mxfp4 .co, but a separate .co built from
// mi350_fmha_hd128_mxfp4_sparse.py with _SORTED_DISPATCH=True (752-byte kernarg + work-table decode).
// Persistent (grid-stride) dispatch is NOT supported for mxfp4 (SGPR budget; see the .py docstring),
// so only the sorted sub-mode exists.
static constexpr const char* kSparseMxfp4SortedCoName =
    "fmha_v3_fwd/fwd_hd128_mxfp4_sparse_sorted.co";
// DENSE mxfp4 sibling (no LUT / no block sparsity): processes every KV tile.
// Built from dmip_asm/fmha_sage_fwd/gfx950/mi350_fmha_hd128_mxfp4.py. It reuses
// the 656-byte fmha_fwd_v3_args layout (init_sparse_v3_args' dense prefix), so
// the same arg-builder feeds it; only the symbol + .co name + 656-byte kernarg
// differ from the sparse path. Same mxfp4 numeric contract (fp4-packed Q/K with
// E8M0 per-block scales applied via MFMA, fp8 V, bf16 out).
static constexpr const char* kMxfp4DenseKernelName =
    "_ZN5aiter28fmha_fwd_hd128_mxfp4_gfx950E";
static constexpr const char* kMxfp4DenseCoName =
    "fmha_v3_fwd/fwd_hd128_mxfp4.co";
// fp8-quantized sibling (E4M3 Q/K/V). Same 720-byte kernarg layout and
// same in_bpe=1 byte stride as the i8fp8 path, so init_sparse_v3_args is
// reused unchanged; only the kernel symbol + .co name differ.
static constexpr const char* kSparseFp8KernelName =
    "_ZN5aiter32fmha_fwd_hd128_fp8_sparse_gfx950E";
static constexpr const char* kSparseFp8CoName =
    "fmha_v3_fwd/fwd_hd128_fp8_sparse.co";
// Finer-KV-granularity fork (kTileKV=64; stage 1 keeps kTileQ=256 / 8 waves, so
// the grid + bdx + 720-byte kernarg are identical -- only the symbol + .co
// differ). Built from mi350_fmha_hd128_fp8_sparse_q128kv64.py. The caller must
// build the LUT with BLOCK_N=64 (kv_block_indices in units of 64).
static constexpr const char* kSparseFp8Q128KV64KernelName =
    "_ZN5aiter41fmha_fwd_hd128_fp8_sparse_q128kv64_gfx950E";
static constexpr const char* kSparseFp8Q128KV64CoName =
    "fmha_v3_fwd/fwd_hd128_fp8_sparse_q128kv64.co";
// Persistent (grid-stride) fp8 sparse sibling. SAME kernel symbol, but a
// separate .co built from mi350_fmha_hd128_fp8_sparse.py with _PERSISTENT=True
// (752-byte kernarg + 1-D grid-stride loop). Kept as a distinct .co so the
// non-persistent path is byte-for-byte unaffected.
static constexpr const char* kSparseFp8PersistentCoName =
    "fmha_v3_fwd/fwd_hd128_fp8_sparse_persistent.co";
// Sorted-dispatch fp8 sparse sibling: one WG per tile, flat grid indexed through
// the LPT work table. Separate .co built with _SORTED_DISPATCH=True.
static constexpr const char* kSparseFp8SortedCoName =
    "fmha_v3_fwd/fwd_hd128_fp8_sparse_sorted.co";

// Pack the 720-byte blob. The first 656 bytes mirror init_fmha_fwd_v3_args
// (see mha_fwd.cu); the trailing 64 bytes hold the 4 LUT pointers (each 16
// bytes with p2 padding, matching the host struct in mha_fwd.h).
//
// We pack manually instead of calling init_fmha_fwd_v3_args to keep the
// sparse path self-contained and avoid mha_fwd.cu's static configs (cfg_fmha_fwd)
// which key on (dtype, hdim, mask, mode, ts_*). The values written here are
// the same ones init_fmha_fwd_v3_args would produce for {data_type="i8fp8bf16",
// is_group_mode=false, mask_type=0, ts_qo=256, in_bpe=1, out_bpe=2}.
static void init_sparse_v3_args(fmha_fwd_v3_sparse_args& args,
                                const mha_fwd_sparse_args& a)
{
    // ---- dense portion (matches mha_fwd.cu::init_fmha_fwd_v3_args path
    // for i8fp8bf16, batch mode, mask=0). ----
    constexpr int in_bpe  = 1;
    constexpr int out_bpe = 2;
    constexpr int ts_qo   = kSparseTileQ;

    args.ptr_o            = a.o_ptr;
    args.ptr_q            = a.q_ptr;
    args.ptr_k            = a.k_ptr;
    args.ptr_v            = a.v_ptr;
    args.ptr_lse          = nullptr;
    args.ptr_qseq         = nullptr;
    args.ptr_kseq         = nullptr;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;
    args.ptr_q_descale    = a.q_descale_ptr;
    args.ptr_k_descale    = a.k_descale_ptr;
    args.ptr_v_descale    = a.v_descale_ptr;
    args.s_descale_q_Bs   = a.batch_stride_q_descale * 4;
    args.s_descale_q_Hs   = a.nhead_stride_q_descale * 4;
    args.s_descale_k_Bs   = a.batch_stride_k_descale * 4;
    args.s_descale_k_Hs   = a.nhead_stride_k_descale * 4;
    args.s_descale_v_Bs   = a.batch_stride_v_descale * 4;
    args.s_descale_v_Hs   = a.nhead_stride_v_descale * 4;

    args.scalar        = a.scale_s;
    args.s_seq_len     = a.seqlen_q;
    args.s_Seqs        = a.stride_q * in_bpe;
    args.s_Ts          = ts_qo * a.stride_q * in_bpe;
    args.s_Hs          = a.nhead_stride_q * in_bpe;
    args.s_Bs          = a.batch_stride_q * in_bpe;
    args.s_gqa         = a.nhead_q / a.nhead_k;
    args.s_k_Seqs      = a.stride_k * in_bpe;
    args.s_k_Hs        = a.nhead_stride_k * in_bpe;
    args.s_k_Bs        = a.batch_stride_k * in_bpe;
    args.s_opt         = 0; // tune_opt unused by the sparse kernel (no mask path)
    args.s_lse         = 0;
    args.s_kv_seq_len  = a.seqlen_k;
    args.s_qk_head_dim = a.hdim_q;
    args.s_v_head_dim  = a.hdim_v;
    args.s_q_head_num  = a.nhead_q;
    args.s_v_Seqs      = a.stride_v * in_bpe;
    args.s_v_Hs        = a.nhead_stride_v * in_bpe;
    args.s_v_Bs        = a.batch_stride_v * in_bpe;
    args.s_o_Seqs      = a.stride_o * out_bpe;
    args.s_o_Hs        = a.nhead_stride_o * out_bpe;
    args.s_o_Bs        = a.batch_stride_o * out_bpe;
    args.s_lse_Hs      = 0;

    // ---- sparse-specific tail ----
    args.ptr_kv_block_indices = a.kv_block_indices_ptr;
    args.ptr_lut_start        = a.lut_start_ptr;
    args.ptr_lut_count        = a.lut_count_ptr;
    // VSA freeze LUT (nullptr => kernel disables freezing == plain sparse).
    args.ptr_lut_freeze       = a.lut_freeze_ptr;
}

float fmha_fwd_v3_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s)
{
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_sparse: only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    if(a.data_type != "i8fp8bf16")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_sparse: only data_type=i8fp8bf16 is "
                          "supported (got " << a.data_type << ")");
        return -1;
    }
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING("fmha_fwd_v3_sparse: unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kSparseKernelName,
        [&]() { return AiterAsmKernel(kSparseKernelName, kSparseCoName); });

    fmha_fwd_v3_sparse_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a);

    // Grid: (num_q_blocks, nhead_q, batch). Same ordering the .py kernel
    // uses for tgid_x/y/z (see process_current_work_sparse()).
    const int num_q_blocks = (a.seqlen_q + kSparseTileQ - 1) / kSparseTileQ;
    const int gdx = num_q_blocks;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

// Block-sparse mxfp4 fmha sibling. Same 720-byte kernarg blob as the
// i8fp8 sparse path; the only on-device difference is the kernel symbol
// + .co name. The mxfp4 kernel re-computes its own Q/K E8M0 per-block
// scale offsets from _s_KV_cur / _s_seq_len / _s_q_head_num so the
// init_sparse_v3_args path can be reused unchanged -- the kernel does
// NOT consume args.s_descale_*_Bs / _Hs for mxfp4 (only the base
// pointers q_descale_ptr / k_descale_ptr / v_descale_ptr matter, which
// init_sparse_v3_args sets from a.{q,k,v}_descale_ptr).
float fmha_fwd_v3_mxfp4_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s)
{
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_mxfp4_sparse: only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    // Accept any caller-tag; we keep the dtype tag opaque to the host so
    // the same dispatcher works whether the wrapper says "mxfp4fp8bf16",
    // "mxfp4bf16", etc. The validation that Q/K really are fp4-packed
    // happens in the torch entry (asm_mha_fwd_sparse.cu::fmha_v3_fwd_mxfp4_sparse).
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING("fmha_fwd_v3_mxfp4_sparse: unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kSparseMxfp4KernelName,
        [&]() { return AiterAsmKernel(kSparseMxfp4KernelName, kSparseMxfp4CoName); });

    fmha_fwd_v3_sparse_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a);

    const int num_q_blocks = (a.seqlen_q + kSparseTileQ - 1) / kSparseTileQ;
    const int gdx = num_q_blocks;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

// DENSE mxfp4 dispatcher (no block sparsity). Reuses init_sparse_v3_args to fill
// the shared 656-byte fmha_fwd_v3_args prefix (the trailing LUT pointers it also
// writes are simply not sent: arg_size is clamped to the dense kernarg size, and
// the dense .co only declares a 656-byte kernarg). Grid + bdx match the sparse
// kernel: (num_q_blocks, nhead_q, batch) with bdx=512. Routes to fwd_hd128_mxfp4.co.
float fmha_fwd_v3_mxfp4(mha_fwd_sparse_args a, const ck_tile::stream_config& s)
{
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_mxfp4: only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING("fmha_fwd_v3_mxfp4: unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kMxfp4DenseKernelName,
        [&]() { return AiterAsmKernel(kMxfp4DenseKernelName, kMxfp4DenseCoName); });

    fmha_fwd_v3_sparse_args args{};
    init_sparse_v3_args(args, a);
    // The dense .co declares a 656-byte kernarg (no LUT tail). Send exactly that
    // many bytes; the (null) LUT pointers init_sparse_v3_args appended are dropped.
    size_t arg_size = sizeof(fmha_fwd_v3_args);

    const int num_q_blocks = (a.seqlen_q + kSparseTileQ - 1) / kSparseTileQ;
    const int gdx = num_q_blocks;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

// Sorted-dispatch mxfp4 sparse dispatcher. One workgroup per tile on a flat 1-D grid
// (gridDim=(total_tiles,1,1)); each WG reads work_table[wg_id] (host-built, LPT-sorted heaviest-first)
// and decodes its (q,h,b) tile. The hardware scheduler stays work-conserving; the table only fixes
// dispatch ORDER so the few heavy tiles spread across CUs and the E2E tail collapses. Requires
// a.work_table_ptr + a.total_tiles. Unlike fp8 there is NO persistent (grid-stride) sub-mode -- mxfp4
// has no free SGPRs for the persistent tile cursor (see mi350_fmha_hd128_mxfp4_sparse.py). Routes to
// fwd_hd128_mxfp4_sparse_sorted.co (752-byte kernarg).
float fmha_fwd_v3_mxfp4_sparse_sorted(mha_fwd_sparse_args a,
                                      const ck_tile::stream_config& s)
{
    const char* tag = "fmha_fwd_v3_mxfp4_sparse_sorted";
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING(tag << ": only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING(tag << ": unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }
    if(a.work_table_ptr == nullptr || a.total_tiles == 0)
    {
        AITER_LOG_WARNING(tag << ": work_table_ptr/total_tiles must be set");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kSparseMxfp4SortedCoName,
        [&]() { return AiterAsmKernel(kSparseMxfp4KernelName, kSparseMxfp4SortedCoName); });

    fmha_fwd_v3_sparse_persistent_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a); // fills the shared 720-byte base (incl. lut_freeze)
    args.ptr_work_table = a.work_table_ptr;
    args.s_total_tiles  = a.total_tiles;
    args.s_num_wgs      = a.total_tiles; // unused by the sorted .co, kept for parity

    const int gdx = static_cast<int>(a.total_tiles);
    const int gdy = 1;
    const int gdz = 1;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

// Block-sparse fp8 fmha sibling (E4M3 Q/K and fp8 V, per-tensor fp32
// descales -- same descale contract as the i8fp8 path). Shares the
// identical 720-byte kernarg blob and in_bpe=1 byte stride as the i8fp8
// path, so init_sparse_v3_args is reused unchanged. The only on-device
// difference is the kernel symbol + .co name.
float fmha_fwd_v3_fp8_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s,
                             bool q128kv64)
{
    const char* tag = q128kv64 ? "fmha_fwd_v3_fp8_sparse[q128kv64]"
                               : "fmha_fwd_v3_fp8_sparse";
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING(tag << ": only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    if(a.data_type != "fp8bf16")
    {
        AITER_LOG_WARNING(tag << ": only data_type=fp8bf16 is "
                          "supported (got " << a.data_type << ")");
        return -1;
    }
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING(tag << ": unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    // Stage 1 of the q128kv64 fork shares the 720-byte kernarg, the (256-Q) grid
    // and bdx=512 (8 waves) with the default kernel; only the symbol + .co differ.
    const char* kname  = q128kv64 ? kSparseFp8Q128KV64KernelName : kSparseFp8KernelName;
    const char* coname = q128kv64 ? kSparseFp8Q128KV64CoName     : kSparseFp8CoName;

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kname,
        [&]() { return AiterAsmKernel(kname, coname); });

    fmha_fwd_v3_sparse_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a);

    const int num_q_blocks = (a.seqlen_q + kSparseTileQ - 1) / kSparseTileQ;
    const int gdx = num_q_blocks;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

// Persistent (grid-stride) fp8 sparse dispatcher. Instead of one workgroup per
// (b, h, q_block) tile, launch a FIXED 1-D grid of num_wgs workgroups; each WG
// grid-strides over a.work_table (host-built, LPT-sorted) so the few heavy tiles
// spread across CUs and the E2E tail-block collapses. Requires a.work_table_ptr
// and a.total_tiles; num_wgs auto-sizes to the CU count when a.num_wgs == 0.
float fmha_fwd_v3_fp8_sparse_persistent(mha_fwd_sparse_args a,
                                        const ck_tile::stream_config& s,
                                        bool sorted_dispatch)
{
    const char* tag = sorted_dispatch ? "fmha_fwd_v3_fp8_sparse_sorted"
                                       : "fmha_fwd_v3_fp8_sparse_persistent";
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING(tag << ": only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    if(a.data_type != "fp8bf16")
    {
        AITER_LOG_WARNING(tag << ": only data_type=fp8bf16 is supported (got "
                          << a.data_type << ")");
        return -1;
    }
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING(tag << ": unsupported feature combination "
                          "(group/mask/lse/dropout/bias must be off)");
        return -1;
    }
    if(a.work_table_ptr == nullptr || a.total_tiles == 0)
    {
        AITER_LOG_WARNING(tag << ": work_table_ptr/total_tiles must be set");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    const char* co_name =
        sorted_dispatch ? kSparseFp8SortedCoName : kSparseFp8PersistentCoName;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        co_name,
        [&]() { return AiterAsmKernel(kSparseFp8KernelName, co_name); });

    fmha_fwd_v3_sparse_persistent_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a); // fills the shared 720-byte base
    args.ptr_work_table = a.work_table_ptr;
    args.s_total_tiles  = a.total_tiles;

    int gdx, gdy = 1, gdz = 1;
    if(sorted_dispatch)
    {
        // One WG per tile; the hardware scheduler stays work-conserving and the
        // LPT-sorted table only fixes dispatch order. num_wgs unused by this .co.
        args.s_num_wgs = a.total_tiles;
        gdx = static_cast<int>(a.total_tiles);
    }
    else
    {
        // Persistent grid-stride: one WG per CU (full occupancy at 1 WG/CU),
        // capped so we never launch more WGs than tiles.
        int num_wgs = static_cast<int>(a.num_wgs);
        if(num_wgs <= 0)
        {
            int dev = 0;
            hipGetDevice(&dev);
            hipDeviceProp_t props{};
            hipGetDeviceProperties(&props, dev);
            num_wgs = props.multiProcessorCount; // CU count
        }
        if(num_wgs > static_cast<int>(a.total_tiles))
            num_wgs = static_cast<int>(a.total_tiles);
        if(num_wgs < 1)
            num_wgs = 1;
        args.s_num_wgs = static_cast<uint32_t>(num_wgs);
        gdx = num_wgs;
    }
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

} // namespace aiter
