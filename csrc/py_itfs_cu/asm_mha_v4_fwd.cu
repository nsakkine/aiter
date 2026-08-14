// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <cstddef>
#include <cstring>

#include "aiter_hip_common.h"
#include "asm_fmha_v4_fwd_configs.hpp"
#include "py_itfs_common.h"
#include "torch/mha_v4_fwd.h"

namespace aiter {
namespace torch_itfs {
namespace {

enum class AttentionFormat : int64_t
{
    Fp32        = 0,
    Fp16        = 1,
    Bf16        = 2,
    Fp8E4M3     = 3,
    Fp8E4M3Fnuz = 4,
    Fp8E5M2     = 5,
    Fp8E5M2Fnuz = 6,
    Fp6E2M3     = 7,
    Fp6E3M2     = 8,
    Fp4E2M1     = 9,
    Int8        = 10,
    UInt8       = 11,
    Int4        = 12,
    UInt4       = 13,
};

constexpr int64_t format_id(AttentionFormat format) { return static_cast<int64_t>(format); }

enum class AttentionScaleMode : int64_t
{
    None                 = 0,
    F32PerTensor         = 1,
    F32PerHead           = 2,
    F32PerToken          = 3,
    F32PerChannel        = 4,
    E8M0Per1x32          = 5,
};

constexpr int64_t scale_mode_id(AttentionScaleMode mode) { return static_cast<int64_t>(mode); }

// Sparse selection is an explicit dispatch dimension, never an inference from extra pointers on a
// dense request. PooledCorrection is its own mode rather than a flag on BlockLut because Sol-Attn
// (arXiv 2607.24027) recovers the skipped blocks' contribution from pooled K/V and so carries three
// more operands in its ABI.
enum class AttentionSparseMode : int64_t
{
    None             = 0,
    BlockLut         = 1,
    PooledCorrection = 2,
};

constexpr int64_t sparse_mode_id(AttentionSparseMode mode) { return static_cast<int64_t>(mode); }

// The one geometry the Sol-Attn kernel is built for. Not tunable knobs: ts_kv is the pooling block
// size and its log2 is folded into the kernel's softmax bias as a constant, so pooling with any
// other value is silently wrong rather than an error. These must match the ts_qo / ts_kv columns of
// the kernel's manifest row.
constexpr int64_t kSolAttnTsQo = 256;
constexpr int64_t kSolAttnTsKv = 128;

constexpr int64_t kHeadDim = 128;

struct PointerSlot
{
    void* value;
    uint32_t padding[2];
};

struct ConstPointerSlot
{
    const void* value;
    uint32_t padding[2];
};

struct ScalarSlot
{
    uint32_t value;
    uint32_t padding[3];
};

struct __attribute__((packed)) FmhaV4Kernarg
{
    PointerSlot ptr_o;
    ConstPointerSlot ptr_q;
    ConstPointerSlot ptr_k;
    ConstPointerSlot ptr_v;
    PointerSlot ptr_lse;
    ScalarSlot scalar;
    ScalarSlot s_seq_len;
    ScalarSlot s_Seqs;
    ScalarSlot s_Ts;
    ScalarSlot s_Hs;
    ScalarSlot s_Bs;
    ScalarSlot s_gqa;
    ScalarSlot s_k_Seqs;
    ScalarSlot s_k_Hs;
    ScalarSlot s_k_Bs;
    ScalarSlot s_opt;
    ScalarSlot s_lse;
    ScalarSlot s_kv_seq_len;
    ScalarSlot s_qk_head_dim;
    ScalarSlot s_v_head_dim;
    ScalarSlot s_q_head_num;
    ScalarSlot s_v_Seqs;
    ScalarSlot s_v_Hs;
    ScalarSlot s_v_Bs;
    ScalarSlot s_o_Seqs;
    ScalarSlot s_o_Hs;
    ScalarSlot s_o_Bs;
    // Reserved v1 slots keep existing dense code objects at their 656-byte ABI. Sparse, varlen,
    // and LSE support may assign them in later manifest rows; current dense rows leave them zero.
    ConstPointerSlot ptr_qseq;
    ConstPointerSlot ptr_kseq;
    ScalarSlot s_lse_Hs;
    ConstPointerSlot ptr_qseq_padding;
    ConstPointerSlot ptr_kseq_padding;
    ConstPointerSlot ptr_q_descale;
    ConstPointerSlot ptr_k_descale;
    ConstPointerSlot ptr_v_descale;
    ScalarSlot s_descale_q_Bs;
    ScalarSlot s_descale_q_Hs;
    ScalarSlot s_descale_k_Bs;
    ScalarSlot s_descale_k_Hs;
    ScalarSlot s_descale_v_Bs;
    ScalarSlot s_descale_v_Hs;
};

static_assert(sizeof(FmhaV4Kernarg) == 656, "MHA v4 dense kernarg ABI must remain 656 bytes");
static_assert(offsetof(FmhaV4Kernarg, ptr_o) == 0x000);
static_assert(offsetof(FmhaV4Kernarg, ptr_q) == 0x010);
static_assert(offsetof(FmhaV4Kernarg, ptr_k) == 0x020);
static_assert(offsetof(FmhaV4Kernarg, ptr_v) == 0x030);
static_assert(offsetof(FmhaV4Kernarg, scalar) == 0x050);
static_assert(offsetof(FmhaV4Kernarg, ptr_q_descale) == 0x200);
static_assert(offsetof(FmhaV4Kernarg, ptr_k_descale) == 0x210);
static_assert(offsetof(FmhaV4Kernarg, ptr_v_descale) == 0x220);

// Sparse kernarg: the dense 656-byte prologue is unchanged and only the 256 bytes from 0x290 on are
// the sparse variant's, so a sparse code object runs the same base ABI as a dense one. The tail is a
// FROZEN contract with the pyisa kernels under ASM/fmha_sage_fwd/gfx950/, which hard-code these
// offsets as s_load literals; reordering anything below must be mirrored in every sparse prologue.
//
//   0x290..0x2B0  ragged LUT: kv_block_indices, lut_start, lut_count
//   0x2C0         ptr_lut_freeze   int32[work items] or null (VSA only)
//   0x2D0         ptr_work_table   uint32[tiles] or null (sorted dispatch)
//   0x2E0..0x300  Sol-Attn pooled keys, pooled values, selection bitmap
//   0x310..0x360  pooled K / V strides, BYTES
//   0x370..0x380  num_kv_blocks, bitmap words per work item
struct __attribute__((packed)) FmhaV4SparseKernarg
{
    FmhaV4Kernarg base;

    ConstPointerSlot ptr_kv_block_indices;
    ConstPointerSlot ptr_lut_start;
    ConstPointerSlot ptr_lut_count;
    ConstPointerSlot ptr_lut_freeze;
    ConstPointerSlot ptr_work_table;

    ConstPointerSlot ptr_mean_k;
    ConstPointerSlot ptr_mean_v;
    ConstPointerSlot ptr_block_bitmap;
    ScalarSlot s_mean_k_Seqs;
    ScalarSlot s_mean_k_Hs;
    ScalarSlot s_mean_k_Bs;
    ScalarSlot s_mean_v_Seqs;
    ScalarSlot s_mean_v_Hs;
    ScalarSlot s_mean_v_Bs;
    ScalarSlot s_num_kv_blocks;
    ScalarSlot s_bitmap_Ds;
};

static_assert(sizeof(FmhaV4SparseKernarg) == 912, "MHA v4 sparse kernarg ABI must remain 912 bytes");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_kv_block_indices) == 0x290, "LUT ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_lut_start) == 0x2A0, "LUT ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_lut_count) == 0x2B0, "LUT ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_lut_freeze) == 0x2C0, "LUT ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_work_table) == 0x2D0, "work-table ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_mean_k) == 0x2E0, "Sol-Attn ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_mean_v) == 0x2F0, "Sol-Attn ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, ptr_block_bitmap) == 0x300, "Sol-Attn ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, s_mean_k_Seqs) == 0x310, "Sol-Attn ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, s_mean_v_Seqs) == 0x340, "Sol-Attn ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, s_num_kv_blocks) == 0x370, "Sol-Attn ABI drift");
static_assert(offsetof(FmhaV4SparseKernarg, s_bitmap_Ds) == 0x380, "Sol-Attn ABI drift");

void check_format_tensor(const at::Tensor& tensor, int64_t format, const char* name)
{
    if(format == format_id(AttentionFormat::Int8))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Char, name, " must be int8");
    }
    else if(format == format_id(AttentionFormat::Fp8E4M3))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float8_e4m3fn,
                    name,
                    " must be FP8 E4M3 FN");
    }
    else if(format == format_id(AttentionFormat::Fp8E4M3Fnuz))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float8_e4m3fnuz,
                    name,
                    " must be FP8 E4M3 FNUZ");
    }
    else if(format == format_id(AttentionFormat::Fp6E2M3) ||
            format == format_id(AttentionFormat::Fp4E2M1))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Byte,
                    name,
                    " must be a uint8 packed MX tensor");
    }
    else
    {
        TORCH_CHECK(false, "unsupported MHA v4 format id: ", format);
    }
}

// Pooled-operand format and scale mode select a row rather than being implied by K/V. Pooling can
// inherit K's descale only because mean(x) * descale == mean(x * descale) holds for a per-tensor
// scale; a block-granular scale image spans several scale blocks per pooled row and has no single
// descale to inherit, so a future MX row must be able to say it pools into some other format.
// Dense rows carry 0 in both columns, where there are no pooled operands at all.
const fmha_v4_fwdConfig& find_config(const std::string& arch,
                                     int64_t q_format,
                                     int64_t k_format,
                                     int64_t v_format,
                                     int64_t q_scale_mode,
                                     int64_t k_scale_mode,
                                     int64_t v_scale_mode,
                                     int64_t mean_format,
                                     int64_t mean_scale_mode,
                                     int64_t sparse_mode)
{
    for(const auto& entry : cfg_fmha_v4_fwd)
    {
        const auto& cfg = entry.second;
        if(cfg.arch == arch && cfg.q_format == q_format && cfg.k_format == k_format &&
               cfg.v_format == v_format && cfg.q_scale_mode == q_scale_mode &&
               cfg.k_scale_mode == k_scale_mode && cfg.v_scale_mode == v_scale_mode &&
               cfg.mean_format == mean_format && cfg.mean_scale_mode == mean_scale_mode &&
               cfg.sparse == sparse_mode &&
               cfg.o_format == format_id(AttentionFormat::Bf16) &&
               cfg.o_scale_mode == scale_mode_id(AttentionScaleMode::None) &&
           cfg.hdim_q == kHeadDim && cfg.hdim_v == kHeadDim && cfg.mask == 0 && cfg.mode == 0)
            return cfg;
    }
    TORCH_CHECK(false,
                "no MHA v4 kernel for arch=",
                arch,
                ", q_format=",
                q_format,
                ", k_format=",
                k_format,
                ", v_format=",
                v_format,
                ", q_scale_mode=",
                q_scale_mode,
                ", k_scale_mode=",
                k_scale_mode,
                ", v_scale_mode=",
                v_scale_mode,
                ", mean_format=",
                mean_format,
                ", mean_scale_mode=",
                mean_scale_mode,
                ", sparse_mode=",
                sparse_mode,
                ", output=BF16, head_dim=128, non-causal batch mode");
}

void set_descale_strides(const at::Tensor& tensor,
                         int head_dimension,
                         uint32_t& batch_stride,
                         uint32_t& head_stride)
{
    if(tensor.dim() >= 2)
    {
        batch_stride = tensor.stride(0) * tensor.element_size();
        head_stride  = tensor.stride(head_dimension) * tensor.element_size();
    }
}

} // namespace

void fmha_v4_fwd(const at::Tensor& q,
                 const at::Tensor& k,
                 const at::Tensor& v,
                 const at::Tensor& q_descale,
                 const at::Tensor& k_descale,
                 const at::Tensor& v_descale,
                 at::Tensor out,
                 int64_t q_format,
                 int64_t k_format,
                 int64_t v_format,
                 int64_t q_scale_mode,
                 int64_t k_scale_mode,
                 int64_t v_scale_mode,
                 double softmax_scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
                "Q, K, V, and out must be GPU tensors");
    TORCH_CHECK(q_descale.is_cuda() && k_descale.is_cuda() && v_descale.is_cuda(),
                "all descale tensors must be GPU tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == out.device(),
                "Q, K, V, and out must be on the same GPU");
    TORCH_CHECK(q_descale.device() == q.device() && k_descale.device() == q.device() &&
                    v_descale.device() == q.device(),
                "all descale tensors must be on the same GPU as Q");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
                "MHA v4 expects BSHD tensors");
    TORCH_CHECK(q_format == k_format, "MHA v4 currently requires matching Q/K formats");
    check_format_tensor(q, q_format, "Q");
    check_format_tensor(k, k_format, "K");
    check_format_tensor(v, v_format, "V");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 &&
                    out.stride(-1) == 1,
                "Q, K, V, and out must have contiguous last dimensions");

    const int64_t batch        = q.size(0);
    const int64_t seqlen_q     = q.size(1);
    const int64_t nhead_q      = q.size(2);
    const int64_t seqlen_k     = k.size(1);
    const int64_t nhead_k      = k.size(2);
    const int64_t packed_width = q_format == format_id(AttentionFormat::Fp6E2M3) ? 96 :
                                 q_format == format_id(AttentionFormat::Fp4E2M1) ? 64 : 128;

    TORCH_CHECK(batch > 0 && seqlen_q > 0 && seqlen_k > 0 && nhead_q > 0,
                "MHA v4 requires non-empty inputs");
    TORCH_CHECK(k.size(0) == batch && v.size(0) == batch, "Q, K, and V batch sizes must match");
    TORCH_CHECK(nhead_q == nhead_k && v.size(2) == nhead_k,
                "MHA v4 initially supports MHA only; Q and KV heads must match");
    TORCH_CHECK(k.size(1) == v.size(1), "K and V sequence lengths must match");
    TORCH_CHECK(q.size(3) == packed_width && k.size(3) == packed_width,
                "Q/K packed width does not match the explicit format");
    TORCH_CHECK(v.size(3) == kHeadDim, "V must have logical head dimension 128");
    if(q_format == format_id(AttentionFormat::Fp4E2M1))
    {
        if(v_format == format_id(AttentionFormat::Fp8E4M3) ||
           v_format == format_id(AttentionFormat::Fp8E4M3Fnuz))
        {
            const int64_t tiles       = (seqlen_k + 127) / 128;
            const int64_t head_stride = tiles * 8192;
            TORCH_CHECK(k.stride(0) == nhead_k * head_stride && k.stride(1) == 64 &&
                            k.stride(2) == head_stride,
                        "MXFP4/FP8 K must use the coalesced MHA v4 tile layout");
        }
        else
        {
            const int64_t tiles       = (seqlen_k + 127) / 128;
            const int64_t head_stride = tiles * 8192;
            TORCH_CHECK(k.stride(0) == nhead_k * head_stride && k.stride(1) == 64 &&
                            k.stride(2) == head_stride,
                        "F4F4 K must use the coalesced MHA v4 tile layout");
        }
    }
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
                "MHA v4 currently supports BF16 output only");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({batch, seqlen_q, nhead_q, kHeadDim}),
                "out must have shape [batch, query_length, query_heads, 128]");

    const bool mx_qk = q_format == format_id(AttentionFormat::Fp6E2M3) ||
                       q_format == format_id(AttentionFormat::Fp4E2M1);
    if(mx_qk)
    {
        TORCH_CHECK(q_descale.scalar_type() == at::ScalarType::Byte &&
                        k_descale.scalar_type() == at::ScalarType::Byte,
                    "MX Q/K descales must be uint8 E8M0 tensors");
        TORCH_CHECK(q_descale.sizes() == torch::IntArrayRef({batch, seqlen_q, nhead_q, 4}),
                    "MX Q descale must have shape [batch, query_length, query_heads, 4]");
        TORCH_CHECK(k_descale.sizes() == torch::IntArrayRef({batch, seqlen_k, nhead_k, 4}),
                    "MX K descale must have shape [batch, key_length, key_heads, 4]");
    }
    else
    {
        TORCH_CHECK(q_descale.scalar_type() == at::ScalarType::Float &&
                        k_descale.scalar_type() == at::ScalarType::Float,
                    "INT8/FP8 Q/K descales must be float32 tensors");
        TORCH_CHECK(q_descale.numel() == 1 && k_descale.numel() == 1,
                    "INT8/FP8 Q/K descales must be scalar tensors");
    }
    const bool mxfp4_v = v_format == format_id(AttentionFormat::Fp4E2M1);
    if(mx_qk && mxfp4_v)
    {
        const int64_t tiles = (seqlen_k + 127) / 128;
        TORCH_CHECK(v_scale_mode == 5 && v_descale.scalar_type() == at::ScalarType::Byte,
                    "MXFP4 V descale must use uint8 E8M0 per-1x32 scales");
        TORCH_CHECK(v_descale.sizes() == torch::IntArrayRef({batch, nhead_k, tiles * 512}),
                    "MXFP4 V descale must have shape [batch, key_heads, tiles * 512]");
    }
    else if(mx_qk)
    {
        TORCH_CHECK(v_descale.scalar_type() == at::ScalarType::Float,
                    "MX FP8 V descale must be a float32 tensor");
        TORCH_CHECK(v_descale.sizes() == torch::IntArrayRef({batch, nhead_k, kHeadDim}),
                    "MX V descale must have shape [batch, key_heads, 128]");
    }
    else
    {
        TORCH_CHECK(v_descale.scalar_type() == at::ScalarType::Float,
                    "INT8/FP8 V descale must be a float32 tensor");
        TORCH_CHECK(v_descale.numel() == 1, "INT8/FP8 V descale must be a scalar tensor");
    }

    const auto arch = get_gpu_arch();
    const auto& cfg = find_config(arch,
                                  q_format,
                                  k_format,
                                  v_format,
                                  q_scale_mode,
                                  k_scale_mode,
                                  v_scale_mode,
                                  0, // dense rows have no pooled operands
                                  0,
                                  sparse_mode_id(AttentionSparseMode::None));

    FmhaV4Kernarg args{};
    args.ptr_o.value         = out.data_ptr();
    args.ptr_q.value         = q.data_ptr();
    args.ptr_k.value         = k.data_ptr();
    args.ptr_v.value         = v.data_ptr();
    args.ptr_q_descale.value = q_descale.data_ptr();
    args.ptr_k_descale.value = k_descale.data_ptr();
    args.ptr_v_descale.value = v_descale.data_ptr();
    static_assert(sizeof(float) == sizeof(uint32_t));
    const float scale = static_cast<float>(softmax_scale);
    std::memcpy(&args.scalar.value, &scale, sizeof(scale));
    args.s_seq_len.value     = seqlen_q;
    args.s_Seqs.value        = q.stride(1);
    args.s_Ts.value          = cfg.ts_qo * q.stride(1);
    args.s_Hs.value          = q.stride(2);
    args.s_Bs.value          = q.stride(0);
    args.s_gqa.value         = 1; // Initial v4 rows are MHA-only.
    args.s_k_Seqs.value      = k.stride(1);
    args.s_k_Hs.value        = k.stride(2);
    args.s_k_Bs.value        = k.stride(0);
    args.s_opt.value         = 5; // Dense, non-causal v1 tuning mode inherited by these binaries.
    args.s_lse.value         = 0;
    args.s_kv_seq_len.value  = seqlen_k;
    args.s_qk_head_dim.value = kHeadDim;
    args.s_v_head_dim.value  = kHeadDim;
    args.s_q_head_num.value  = nhead_q;
    args.s_v_Seqs.value      = v.stride(1);
    args.s_v_Hs.value        = v.stride(2);
    args.s_v_Bs.value        = v.stride(0);
    // Input tensors are byte-addressed packed formats, so their element strides already equal
    // byte strides. BF16 output strides require the explicit two-byte conversion.
    args.s_o_Seqs.value      = out.stride(1) * 2;
    args.s_o_Hs.value        = out.stride(2) * 2;
    args.s_o_Bs.value        = out.stride(0) * 2;

    set_descale_strides(
        q_descale,
        q_descale.dim() >= 3 ? 2 : 1,
        args.s_descale_q_Bs.value,
        args.s_descale_q_Hs.value);
    set_descale_strides(
        k_descale,
        k_descale.dim() >= 3 ? 2 : 1,
        args.s_descale_k_Bs.value,
        args.s_descale_k_Hs.value);
    // Production V descales are [batch, head, channel], so the head dimension is 1.
    set_descale_strides(v_descale,
                        1,
                        args.s_descale_v_Bs.value,
                        args.s_descale_v_Hs.value);

    static SynchronizedCache<std::string, AiterAsmKernel> kernels;
    const std::string cache_key = arch + "|" + cfg.knl_name + "|" + cfg.co_name;
    auto& kernel = kernels.get_or_create(cache_key, [&]() {
        return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str());
    });

    size_t arg_size = sizeof(args);
    const int gdx   = (seqlen_q + cfg.ts_qo - 1) / cfg.ts_qo;
    const int gdy   = nhead_q;
    const int gdz   = batch;
    const HipDeviceGuard device_guard{q.get_device()};
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    kernel.launch_kernel({&args, &arg_size, gdx, gdy, gdz, 512, 1, 1, stream});
}

void fmha_v4_fwd_sparse(const at::Tensor& q,
                        const at::Tensor& k,
                        const at::Tensor& v,
                        const at::Tensor& q_descale,
                        const at::Tensor& k_descale,
                        const at::Tensor& v_descale,
                        const at::Tensor& mean_k,
                        const at::Tensor& mean_v,
                        const at::Tensor& kv_block_indices,
                        const at::Tensor& lut_start,
                        const at::Tensor& lut_count,
                        const at::Tensor& block_bitmap,
                        at::Tensor out,
                        int64_t q_format,
                        int64_t k_format,
                        int64_t v_format,
                        int64_t q_scale_mode,
                        int64_t k_scale_mode,
                        int64_t v_scale_mode,
                        int64_t mean_format,
                        int64_t mean_scale_mode,
                        int64_t sparse_mode,
                        double softmax_scale)
{
    // The routing operands come from aiter.ops.triton.attention.utils.sol_attn_prepare, which is
    // what establishes the host contracts this launcher cannot check: the bitmap agreeing with the
    // LUT bit for bit, a selected partial tail block, set padding bits, and lut_count >= 1 for
    // every work item. Reading any of those here would force a device sync on every call.
    TORCH_CHECK(sparse_mode == sparse_mode_id(AttentionSparseMode::PooledCorrection),
                "MHA v4 implements sparse mode ",
                sparse_mode_id(AttentionSparseMode::PooledCorrection),
                " (pooled correction) only, got ",
                sparse_mode);
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
                "Q, K, V, and out must be GPU tensors");
    TORCH_CHECK(q_descale.is_cuda() && k_descale.is_cuda() && v_descale.is_cuda(),
                "all descale tensors must be GPU tensors");
    TORCH_CHECK(mean_k.is_cuda() && mean_v.is_cuda() && block_bitmap.is_cuda() &&
                    kv_block_indices.is_cuda() && lut_start.is_cuda() && lut_count.is_cuda(),
                "all pooled and routing tensors must be GPU tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == out.device(),
                "Q, K, V, and out must be on the same GPU");
    TORCH_CHECK(mean_k.device() == q.device() && mean_v.device() == q.device() &&
                    block_bitmap.device() == q.device() && lut_start.device() == q.device(),
                "pooled and routing tensors must be on the same GPU as Q");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
                "MHA v4 expects BSHD tensors");
    TORCH_CHECK(mean_k.dim() == 4 && mean_v.dim() == 4,
                "pooled K and V must be BSHD with the sequence replaced by the KV block count");
    TORCH_CHECK(q_format == k_format, "MHA v4 currently requires matching Q/K formats");
    check_format_tensor(q, q_format, "Q");
    check_format_tensor(k, k_format, "K");
    check_format_tensor(v, v_format, "V");
    check_format_tensor(mean_k, mean_format, "pooled K");
    check_format_tensor(mean_v, mean_format, "pooled V");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 &&
                    out.stride(-1) == 1 && mean_k.stride(-1) == 1 && mean_v.stride(-1) == 1,
                "Q, K, V, out, and the pooled tensors must have contiguous last dimensions");
    TORCH_CHECK(kv_block_indices.scalar_type() == at::ScalarType::Int &&
                    lut_start.scalar_type() == at::ScalarType::Int &&
                    lut_count.scalar_type() == at::ScalarType::Int,
                "the ragged LUT tensors must be int32");
    TORCH_CHECK(lut_start.is_contiguous() && lut_count.is_contiguous() &&
                    kv_block_indices.is_contiguous(),
                "the ragged LUT tensors must be contiguous");
    TORCH_CHECK(block_bitmap.scalar_type() == at::ScalarType::UInt32,
                "block_bitmap must be uint32");
    TORCH_CHECK(block_bitmap.is_contiguous(), "block_bitmap must be contiguous");

    const int64_t batch    = q.size(0);
    const int64_t seqlen_q = q.size(1);
    const int64_t nhead_q  = q.size(2);
    const int64_t seqlen_k = k.size(1);
    const int64_t nhead_k  = k.size(2);

    TORCH_CHECK(batch > 0 && seqlen_q > 0 && seqlen_k > 0 && nhead_q > 0,
                "MHA v4 requires non-empty inputs");
    TORCH_CHECK(k.size(0) == batch && v.size(0) == batch, "Q, K, and V batch sizes must match");
    TORCH_CHECK(v.size(2) == nhead_k, "K and V head counts must match");
    TORCH_CHECK(k.size(1) == v.size(1), "K and V sequence lengths must match");
    TORCH_CHECK(nhead_q % nhead_k == 0, "query heads must be a multiple of KV heads");
    // The kernel derives a work item's KV head by SHIFTING its query head right by
    // floor(log2(ratio)), not by dividing, so a non-power-of-two ratio silently reads the wrong KV
    // head for most heads instead of failing.
    const int64_t gqa_ratio = nhead_q / nhead_k;
    TORCH_CHECK((gqa_ratio & (gqa_ratio - 1)) == 0,
                "Sol-Attn requires a power-of-two query:KV head ratio, got ",
                nhead_q,
                ":",
                nhead_k);
    TORCH_CHECK(q.size(3) == kHeadDim && k.size(3) == kHeadDim && v.size(3) == kHeadDim,
                "Sol-Attn supports head dimension 128 only");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
                "MHA v4 currently supports BF16 output only");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({batch, seqlen_q, nhead_q, kHeadDim}),
                "out must have shape [batch, query_length, query_heads, 128]");
    TORCH_CHECK(q_descale.scalar_type() == at::ScalarType::Float &&
                    k_descale.scalar_type() == at::ScalarType::Float &&
                    v_descale.scalar_type() == at::ScalarType::Float,
                "Sol-Attn descales must be float32 tensors");
    TORCH_CHECK(q_descale.numel() == 1 && k_descale.numel() == 1 && v_descale.numel() == 1,
                "Sol-Attn descales must be scalar tensors");
    // The kernel folds K's descale into the softmax temperature and V's into the epilogue, so the
    // pooled tensors have no descale of their own and must have been produced under K's and V's.
    TORCH_CHECK(mean_scale_mode == k_scale_mode && mean_scale_mode == v_scale_mode,
                "pooled operands must share K's and V's scale mode, got ",
                mean_scale_mode,
                " against ",
                k_scale_mode,
                " and ",
                v_scale_mode);

    const int64_t num_q_tiles    = (seqlen_q + kSolAttnTsQo - 1) / kSolAttnTsQo;
    const int64_t num_kv_blocks  = (seqlen_k + kSolAttnTsKv - 1) / kSolAttnTsKv;
    const int64_t num_work_items = batch * nhead_q * num_q_tiles;
    // 4 dwords per 128-block group, not 1 dword per 32 blocks: the kernel reads a whole group per
    // approximate tile with one s_load_dwordx4 at byte offset 16 * tile.
    const int64_t bitmap_Ds = 4 * ((num_kv_blocks + 127) / 128);

    TORCH_CHECK(
        mean_k.sizes() == torch::IntArrayRef({batch, num_kv_blocks, nhead_k, kHeadDim}),
        "pooled K must have shape [batch, kv_blocks, kv_heads, 128]");
    TORCH_CHECK(
        mean_v.sizes() == torch::IntArrayRef({batch, num_kv_blocks, nhead_k, kHeadDim}),
        "pooled V must have shape [batch, kv_blocks, kv_heads, 128]");
    TORCH_CHECK(block_bitmap.sizes() == torch::IntArrayRef({num_work_items, bitmap_Ds}),
                "block_bitmap must have shape [work_items, 4 * ceil(kv_blocks / 128)]");
    TORCH_CHECK(lut_start.sizes() == torch::IntArrayRef({num_work_items}) &&
                    lut_count.sizes() == torch::IntArrayRef({num_work_items}),
                "lut_start and lut_count must have one entry per (batch, query head, query tile)");

    const auto arch = get_gpu_arch();
    const auto& cfg = find_config(arch,
                                  q_format,
                                  k_format,
                                  v_format,
                                  q_scale_mode,
                                  k_scale_mode,
                                  v_scale_mode,
                                  mean_format,
                                  mean_scale_mode,
                                  sparse_mode);
    TORCH_CHECK(cfg.ts_qo == kSolAttnTsQo && cfg.ts_kv == kSolAttnTsKv,
                "the selected Sol-Attn row has tile geometry ",
                cfg.ts_qo,
                "x",
                cfg.ts_kv,
                ", but the pooled operands were built for ",
                kSolAttnTsQo,
                "x",
                kSolAttnTsKv);

    FmhaV4SparseKernarg args{};
    FmhaV4Kernarg& base       = args.base;
    base.ptr_o.value          = out.data_ptr();
    base.ptr_q.value          = q.data_ptr();
    base.ptr_k.value          = k.data_ptr();
    base.ptr_v.value          = v.data_ptr();
    base.ptr_q_descale.value  = q_descale.data_ptr();
    base.ptr_k_descale.value  = k_descale.data_ptr();
    base.ptr_v_descale.value  = v_descale.data_ptr();
    static_assert(sizeof(float) == sizeof(uint32_t));
    const float scale = static_cast<float>(softmax_scale);
    std::memcpy(&base.scalar.value, &scale, sizeof(scale));
    base.s_seq_len.value     = seqlen_q;
    base.s_Seqs.value        = q.stride(1);
    base.s_Ts.value          = cfg.ts_qo * q.stride(1);
    base.s_Hs.value          = q.stride(2);
    base.s_Bs.value          = q.stride(0);
    base.s_gqa.value         = gqa_ratio;
    base.s_k_Seqs.value      = k.stride(1);
    base.s_k_Hs.value        = k.stride(2);
    base.s_k_Bs.value        = k.stride(0);
    base.s_opt.value         = 5; // Non-causal v1 tuning mode, as for the dense rows.
    base.s_lse.value         = 0;
    base.s_kv_seq_len.value  = seqlen_k;
    base.s_qk_head_dim.value = kHeadDim;
    base.s_v_head_dim.value  = kHeadDim;
    base.s_q_head_num.value  = nhead_q;
    base.s_v_Seqs.value      = v.stride(1);
    base.s_v_Hs.value        = v.stride(2);
    base.s_v_Bs.value        = v.stride(0);
    base.s_o_Seqs.value      = out.stride(1) * 2;
    base.s_o_Hs.value        = out.stride(2) * 2;
    base.s_o_Bs.value        = out.stride(0) * 2;

    args.ptr_kv_block_indices.value = kv_block_indices.data_ptr();
    args.ptr_lut_start.value        = lut_start.data_ptr();
    args.ptr_lut_count.value        = lut_count.data_ptr();
    // Reserved: this kernel dispatches in grid order and has no VSA freeze list.
    args.ptr_lut_freeze.value   = nullptr;
    args.ptr_work_table.value   = nullptr;
    args.ptr_mean_k.value       = mean_k.data_ptr();
    args.ptr_mean_v.value       = mean_v.data_ptr();
    args.ptr_block_bitmap.value = block_bitmap.data_ptr();
    // The pooled operands' format is a dispatch dimension of its own, so convert their strides to
    // bytes explicitly rather than relying on K/V's one-byte elements.
    const int64_t mean_k_bpe = mean_k.element_size();
    const int64_t mean_v_bpe = mean_v.element_size();
    args.s_mean_k_Seqs.value = mean_k.stride(1) * mean_k_bpe;
    args.s_mean_k_Hs.value   = mean_k.stride(2) * mean_k_bpe;
    args.s_mean_k_Bs.value   = mean_k.stride(0) * mean_k_bpe;
    args.s_mean_v_Seqs.value = mean_v.stride(1) * mean_v_bpe;
    args.s_mean_v_Hs.value   = mean_v.stride(2) * mean_v_bpe;
    args.s_mean_v_Bs.value   = mean_v.stride(0) * mean_v_bpe;
    args.s_num_kv_blocks.value = num_kv_blocks;
    args.s_bitmap_Ds.value     = bitmap_Ds;

    static SynchronizedCache<std::string, AiterAsmKernel> kernels;
    const std::string cache_key = arch + "|" + cfg.knl_name + "|" + cfg.co_name;
    auto& kernel = kernels.get_or_create(cache_key, [&]() {
        return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str());
    });

    size_t arg_size = sizeof(args);
    const int gdx   = num_q_tiles;
    const int gdy   = nhead_q;
    const int gdz   = batch;
    const HipDeviceGuard device_guard{q.get_device()};
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    kernel.launch_kernel({&args, &arg_size, gdx, gdy, gdz, 512, 1, 1, stream});
}

} // namespace torch_itfs
} // namespace aiter
