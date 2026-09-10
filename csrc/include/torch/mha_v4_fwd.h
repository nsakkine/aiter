#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// Validate packed operands, select the exact format/scale manifest row, and launch its code object.
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
                 double softmax_scale);

// Sorted block-sparse sibling. Same packed operands as fmha_v4_fwd, plus a ragged LUT.
// Builds the work table internally (identity raster if lut_count is uniform, else LPT).
void fmha_v4_fwd_sparse(const at::Tensor& q,
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
                        double softmax_scale,
                        const at::Tensor& kv_block_indices,
                        const at::Tensor& lut_start,
                        const at::Tensor& lut_count);

// Sol-Attn sibling (arXiv 2607.24027): the LUT above still drives an EXACT pass, and a second pass
// then sweeps the pooled per-block K/V, masking off the blocks the LUT already covered via
// block_bitmap, so the below-threshold blocks contribute their zeroth-order term instead of nothing.
// Both passes share one online-softmax state, which is what normalizes the mix under a single L.
//
// mean_k / mean_v are K / V with seqlen -> num_kv_blocks in the SOURCE quantized dtype, so the
// approximate pass reuses k_descale / v_descale unchanged; block_bitmap is uint32
// [batch * query_heads * query_tiles, 4 * ceil(num_kv_blocks / 128)] with the bits at and above
// num_kv_blocks SET, which is what clips the last tile's overhang. aiter.ops.triton's
// sol_attn_prepare() produces all six tensors consistently from one boolean mask.
//
// A row may select nothing. lut_count == 0 leaves the exact pass with no running max, and the
// approximate pass recovers one, so such a row lands on the pooled-only softmax over every block.
//
// Dispatches the dense 3-D grid: threshold routing leaves the per-row block counts near uniform
// (max/mean ~1.1), so sorted dispatch buys ~2-4% of scheduling at more than that in grid overhead,
// and its work_table slot at 0x2D0 collides with this ABI's pooled pointers anyway.
void fmha_v4_fwd_sol_attn(const at::Tensor& q,
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
                          double softmax_scale,
                          const at::Tensor& kv_block_indices,
                          const at::Tensor& lut_start,
                          const at::Tensor& lut_count,
                          const at::Tensor& mean_k,
                          const at::Tensor& mean_v,
                          const at::Tensor& block_bitmap,
                          const std::optional<at::Tensor>& mean_k_scale,
                          const std::optional<at::Tensor>& mean_v_scale);

// The work table fmha_v4_fwd_sparse builds internally, exposed so its ordering can be tested.
// Reordering a permutation costs only load balance, but the table must stay a permutation: each
// entry names the tile one workgroup claims, so a duplicated entry leaves another tile unwritten.
at::Tensor mha_v4_sparse_work_table(const at::Tensor& lut_count,
                                    int64_t batch,
                                    int64_t nhead,
                                    int64_t q_tiles);

} // namespace torch_itfs
} // namespace aiter