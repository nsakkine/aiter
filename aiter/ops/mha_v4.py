# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA v4 preprocessing, packed-layout helpers, and launch APIs.

Raw BF16 BSHD operands are quantized into the layouts consumed by the MHA v4
ASM kernels. Format and scale-mode IDs are part of the launcher ABI. Optional
block-sparse execution uses a boolean tile mask on the raw API and a ragged
LUT triple on the packed API; the work table is built inside the sparse
custom op.
"""

import csv
import functools
import os
from enum import IntEnum
from typing import Optional

import torch
import triton
from torch import Tensor

from aiter import dtypes
from aiter.jit.core import AITER_ROOT_DIR, compile_ops
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    mha_v4_per_tensor_amax_kernel,
    mha_v4_per_tensor_quant_kernel,
    mha_v4_per_tensor_scale_kernel,
    sage_quant_v_amax_finalize_kernel,
    sage_quant_v_amax_partial_kernel,
    sage_quant_v_kernel,
)
from aiter.ops.triton.attention.utils import (
    block_attn_mask_to_ragged_lut,
    sol_attn_prepare,
)
from aiter.ops.triton.quant.mxfp6_fmha_pack import (
    fp6_k_lds_order_views_from_raw,
    fp6_k_raw_buffer_sizes,
    pack_fp6_v_data_scale_views,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    fp4_v_padded_sequence,
    fp4_v_raw_buffer_size,
    pack_v_mxfp4_colmajor_raw,
)

MHA_V4_LOG2E = 1.4426950408889634
MHA_V4_PER_TENSOR_BLOCK_SIZE = 8192


def mha_v4_q_multiplier(softmax_scale: float) -> float:
    """Return the Q multiplier expected by the MX attention quantizers."""
    return softmax_scale * MHA_V4_LOG2E


@compile_ops("module_mha_v4_quant", develop=True)
def rotate_activation_hd128(out: Tensor, input: Tensor) -> None:
    """Apply normalized Walsh-Hadamard rotation to contiguous hd128 rows."""


@compile_ops("module_mha_v4_quant", develop=True)
def rotate_activation_mxfp8_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and quantize directly to MXFP8."""


@compile_ops("module_mha_v4_quant", develop=True)
def rotate_activation_mxfp6_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack directly to MXFP6 E2M3."""


@compile_ops("module_mha_v4_quant", develop=True)
def rotate_activation_mxfp6_quant_k(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
) -> None:
    """Rotate and pack hd128 K directly into the MXFP6 LDS-order buffers."""


@compile_ops("module_mha_v4_quant", develop=True)
def rotate_activation_mxfp4_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack directly to MXFP4 E2M1."""


@compile_ops("module_mha_v4_quant", develop=True)
def rotate_activation_mxfp4_quant_k(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack K in the MXFP4 ASM tile order."""


def _mha_v4_sparse_work_table_fake(
    lut_count: Tensor,
    batch: int,
    nhead: int,
    q_tiles: int,
) -> Tensor:
    del lut_count
    return torch.empty(batch * nhead * q_tiles, dtype=torch.int32, device="cuda")


@compile_ops("module_fmha_v4_fwd", gen_fake=_mha_v4_sparse_work_table_fake)
def mha_v4_sparse_work_table(
    lut_count: Tensor,
    batch: int,
    nhead: int,
    q_tiles: int,
) -> Tensor:
    """Return the tile visit order the sparse kernel would use for these LUT lengths.

    Exposed for testing. The sparse launcher builds this itself, and a wrong order only unbalances
    the waves rather than changing the result, so no test of the attention output can see it.

    Each entry packs one tile as ``q_tile | head << 16 | batch << 24``, ordered by LUT length
    descending with ties left in raster order.
    """


class AttentionFormat(IntEnum):
    """Stable operand-encoding IDs used by the Python/C++ dispatch ABI.

    MX names are aliases for their underlying element encoding; scale
    granularity is represented independently by :class:`AttentionScaleMode`.
    """

    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8_E4M3 = 3
    FP8_E4M3_FNUZ = 4
    FP8_E5M2 = 5
    FP8_E5M2_FNUZ = 6
    FP6_E2M3 = 7
    FP6_E3M2 = 8
    FP4_E2M1 = 9
    INT8 = 10
    UINT8 = 11
    INT4 = 12
    UINT4 = 13
    # Aliases
    FP8 = FP8_E4M3
    MXFP6_E2M3 = FP6_E2M3
    MXFP6 = FP6_E2M3
    MXFP6_E3M2 = FP6_E3M2
    MXBF6 = FP6_E3M2
    MXFP4 = FP4_E2M1


class AttentionScaleMode(IntEnum):
    """Stable IDs describing how each operand's descale tensor is indexed."""

    NONE = 0
    F32_PER_TENSOR = 1
    F32_PER_HEAD = 2
    F32_PER_TOKEN = 3
    F32_PER_CHANNEL = 4
    E8M0_PER_1X32 = 5


_FP8_FORMATS = (AttentionFormat.FP8_E4M3, AttentionFormat.FP8_E4M3_FNUZ)
_MX_FORMATS = (AttentionFormat.FP6_E2M3, AttentionFormat.FP4_E2M1)
_PACKED_QK_WIDTH = {
    AttentionFormat.BF16: 128,
    AttentionFormat.INT8: 128,
    AttentionFormat.FP8_E4M3: 128,
    AttentionFormat.FP8_E4M3_FNUZ: 128,
    AttentionFormat.FP6_E2M3: 96,
    AttentionFormat.FP4_E2M1: 64,
}

_MHA_V4_Q_TILE = 256
# mode=1 selects the sorted-sparse manifest rows; the launcher dispatches the same rows through
# find_config(..., mode=1).
MHA_V4_SPARSE_MODE = 1
# mode=2 selects the Sol-Attn rows, which run the same block-sparse exact pass and then correct it
# with a pooled approximate pass, so they share the sparse modes' block geometry.
MHA_V4_SOL_ATTN_MODE = 2
MHA_V4_BLOCK_SPARSE_MODES = (MHA_V4_SPARSE_MODE, MHA_V4_SOL_ATTN_MODE)


def native_fp8_format() -> AttentionFormat:
    """Return the FP8 E4M3 encoding native to the active GPU architecture."""
    return (
        AttentionFormat.FP8_E4M3_FNUZ
        if get_gfx() == "gfx942"
        else AttentionFormat.FP8_E4M3
    )


@functools.cache
def mha_v4_kv_tile() -> int:
    """Return the KV tile of the DEFAULT block-sparse MHA v4 geometry on the active GPU.

    Read from the same manifest the launcher dispatches on rather than restated here, so adding a
    sparse row with a different tile cannot leave the two disagreeing. 128 on gfx950, 64 on gfx942.
    """
    return mha_v4_block_tile()[1]


@functools.cache
def mha_v4_block_tile() -> tuple[int, int]:
    """Return the default block-sparse (q_tile, kv_tile) on the active GPU.

    The default is the geometry at _MHA_V4_Q_TILE, which is the one every existing caller's block
    mask is shaped for. An arch may ship others -- gfx950 also has a 64x64 FP8 row, for models
    routed more finely than a 128-token block -- and mha_v4_block_tiles() lists them.
    """
    return (_MHA_V4_Q_TILE, mha_v4_kv_tile_for_q_tile(_MHA_V4_Q_TILE))


@functools.cache
def mha_v4_kv_tile_for_q_tile(q_tile: int, operands=None, mode=None) -> int:
    """Return the KV tile the block-sparse rows serve at `q_tile`, or 0 if none does.

    A Q tile pins the KV tile because the two are one kernel: the geometry is a property of the
    manifest row, not a pair a caller may mix. Returning 0 rather than raising lets a caller offer
    a geometry and fall back, which is the shape the validation in mha_v4_packed() wants.

    Both filters are optional and both narrow what is being asked:

    operands is what mha_v4_operands() builds -- the three formats and their three scale modes,
    which together are what picks a manifest row. It matters wherever the answer is not the default
    geometry, because an arch need not serve every geometry in every precision: gfx950's 64x64 rows
    are FP8 only, so asking without operands says 64x64 exists and asking with MX ones says it does
    not. The scale modes are part of it and not an over-specification, since the FP8 and MXFP8 rows
    take the same three formats and differ only there.

    mode picks one of MHA_V4_BLOCK_SPARSE_MODES instead of answering for all of them. Left None
    the modes are intersected rather than unioned, since a caller shaping a mask has not yet chosen
    how to route it and a geometry is only safe when either mode can serve it. That intersection is
    only meaningful across whole modes, though: the modes need not agree on operands at a given
    geometry (mxfp4 routes sorted-sparse with an FP8 V and Sol-Attn with an MXFP4 one), so a caller
    narrowing by operands has committed to one mode and should name it.
    """
    return _mha_v4_kv_tile_for_q_tile_from_manifest(
        q_tile,
        *(_ANY_OPERANDS if operands is None else operands),
        _ANY_MODE if mode is None else int(mode),
    )


def mha_v4_block_tiles(operands=None, mode=None) -> tuple[tuple[int, int], ...]:
    """Every block-sparse (q_tile, kv_tile) the active GPU has kernels for, ascending.

    Narrowed by `operands` and `mode` as mha_v4_kv_tile_for_q_tile is, which see.

    Unlike the accessors above this reads the manifest unguarded, so it is NOT traceable -- it is
    for choosing a geometry and for error messages, both of which happen outside a compiled region.
    """
    return tuple(
        (q_tile, kv_tile)
        for q_tile, kv_tile in (
            (q_tile, mha_v4_kv_tile_for_q_tile(q_tile, operands, mode))
            for q_tile in _mha_v4_block_q_tiles_from_manifest()
        )
        if kv_tile
    )


def mha_v4_operands(
    q_format, k_format, v_format, q_scale_mode, k_scale_mode, v_scale_mode
) -> tuple[int, ...]:
    """The six values that pick a manifest row, as ints, for the geometry queries above.

    Plain ints rather than the enums so the value is hashable for their caches and carryable
    through the torch.compile guard the manifest read sits behind.
    """
    return (
        int(q_format),
        int(k_format),
        int(v_format),
        int(q_scale_mode),
        int(k_scale_mode),
        int(v_scale_mode),
    )


# Wildcards for the queries above, negative because every real format, scale mode and mode is not:
# 0 is a valid value for all three, so it cannot stand for "unspecified".
_ANY_OPERANDS = (-1,) * 6
_ANY_MODE = -1


# The reads have to sit behind the guard, not just behind the caches above: Dynamo traces the body
# of a cached function regardless of whether the cache is warm, and `open` is not traceable, so an
# unguarded read costs two graph breaks on every mha_v4(block_mask=...) trace and fails outright
# under fullgraph=True. get_gfx() keeps its own rocminfo probe opaque the same way. The guard only
# carries scalars, which is why this is a query per Q tile rather than one call returning the set.
@torch_compile_guard()
def _mha_v4_kv_tile_for_q_tile_from_manifest(
    q_tile: int,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    mode: int,
) -> int:
    wanted = (q_format, k_format, v_format, q_scale_mode, k_scale_mode, v_scale_mode)

    def serves(row):
        return row[_ROW_TS_QO] == q_tile and all(
            want < 0 or want == have for want, have in zip(wanted, row)
        )

    by_mode = _mha_v4_block_rows_from_manifest()
    if mode != _ANY_MODE:
        if mode not in by_mode:
            raise ValueError(
                f"mode={mode} is not block-sparse; MHA v4 geometry is a property of the "
                f"{MHA_V4_BLOCK_SPARSE_MODES} rows"
            )
        by_mode = {mode: by_mode[mode]}
    # Intersected rather than unioned: a caller that named no mode builds one mask and may route it
    # either way, so a geometry is only usable when every mode can serve it.
    kv_tiles = set.intersection(
        *({row[_ROW_TS_KV] for row in rows if serves(row)} for rows in by_mode.values())
    )
    if len(kv_tiles) > 1:
        raise ValueError(
            f"{get_gfx()} block-sparse rows disagree on ts_kv at ts_qo={q_tile} "
            f"({sorted(kv_tiles)}); the mask geometry a caller builds is only well defined when "
            "they agree"
        )
    return kv_tiles.pop() if kv_tiles else 0


def _mha_v4_block_q_tiles_from_manifest() -> tuple[int, ...]:
    by_mode = _mha_v4_block_rows_from_manifest()
    return tuple(sorted({row[_ROW_TS_QO] for rows in by_mode.values() for row in rows}))


# A row as _mha_v4_block_rows_from_manifest stores it: mha_v4_operands()'s six values, then the
# geometry. One order for both, so a filter can zip a query against a row's head.
_ROW_TS_QO, _ROW_TS_KV = 6, 7


@functools.cache
def _mha_v4_block_rows_from_manifest() -> dict[int, frozenset[tuple[int, ...]]]:
    """Block-sparse manifest rows per mode, each as mha_v4_operands() + (ts_qo, ts_kv).

    The operands are carried because a geometry can be precision-specific; see
    mha_v4_kv_tile_for_q_tile.
    """
    gfx = get_gfx()
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    manifest = os.path.join(asm_dir, gfx, "fmha_v4_fwd", "fmha_v4_fwd.csv")
    by_mode: dict[int, set[tuple[int, ...]]] = {
        mode: set() for mode in MHA_V4_BLOCK_SPARSE_MODES
    }
    try:
        with open(manifest, newline="") as handle:
            for row in csv.DictReader(
                filter(lambda line: not line.startswith("#"), handle)
            ):
                mode = int(row["mode"])
                if mode in by_mode:
                    by_mode[mode].add(
                        mha_v4_operands(
                            row["q_format"],
                            row["k_format"],
                            row["v_format"],
                            row["q_scale_mode"],
                            row["k_scale_mode"],
                            row["v_scale_mode"],
                        )
                        + (int(row["ts_qo"]), int(row["ts_kv"]))
                    )
    except FileNotFoundError as error:
        raise ValueError(
            f"no MHA v4 manifest for {gfx} at {manifest}; block-sparse MHA v4 is "
            "unavailable on this GPU"
        ) from error
    if not all(by_mode.values()):
        raise ValueError(
            f"{gfx} has no manifest row for every block-sparse mode "
            f"{MHA_V4_BLOCK_SPARSE_MODES}; per-mode geometries: "
            f"{ {mode: sorted(row[_ROW_TS_QO:] for row in rows) for mode, rows in by_mode.items()} }"
        )
    return {mode: frozenset(rows) for mode, rows in by_mode.items()}


def _is_fp8_format(format: AttentionFormat) -> bool:
    return format in _FP8_FORMATS


def _validate_bshd_hd128(input: Tensor, operation: str) -> tuple[int, int, int, int]:
    if input.dim() != 4 or input.shape[-1] != 128 or not input.is_contiguous():
        raise ValueError(f"{operation} requires contiguous hd128 BSHD input")
    return input.shape


def _validate_format_contract(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
) -> None:
    if q_format == AttentionFormat.FP6_E3M2:
        raise NotImplementedError(
            "FP6 E3M2 has a reserved format ID but no kernel row yet"
        )
    if q_format != k_format:
        raise ValueError("MHA v4 currently requires matching Q and K formats")
    if q_format == AttentionFormat.BF16:
        # BF16 V keeps the whole recipe unquantized; an FP8 V halves the V traffic and the PV
        # operand width while Q/K stay BF16, which is the only row storing its operands at two
        # different element widths.
        if v_format != AttentionFormat.BF16 and not _is_fp8_format(v_format):
            raise ValueError("BF16 Q/K currently requires BF16 or FP8 V")
        return
    if q_format not in _PACKED_QK_WIDTH:
        raise ValueError(f"unsupported Q/K format: {q_format!r}")
    if v_format not in (
        *_FP8_FORMATS,
        AttentionFormat.FP6_E2M3,
        AttentionFormat.FP4_E2M1,
    ):
        raise ValueError(f"unsupported V format: {v_format!r}")
    if q_format == AttentionFormat.INT8 and v_format not in _FP8_FORMATS:
        raise ValueError("INT8 Q/K currently requires FP8 V")
    if q_format in _FP8_FORMATS and v_format not in (
        q_format,
        AttentionFormat.MXFP6,
    ):
        raise ValueError("FP8 Q/K requires matching FP8 or MXFP6 V")


def scale_modes_for_formats(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
) -> tuple[AttentionScaleMode, AttentionScaleMode, AttentionScaleMode]:
    """Return the canonical Q, K, and V scale modes for a format recipe."""
    _validate_format_contract(q_format, k_format, v_format)
    if q_format == AttentionFormat.BF16:
        return (
            AttentionScaleMode.NONE,
            AttentionScaleMode.NONE,
            (
                AttentionScaleMode.F32_PER_TENSOR
                if _is_fp8_format(v_format)
                else AttentionScaleMode.NONE
            ),
        )
    if q_format == AttentionFormat.INT8 or q_format in _FP8_FORMATS:
        v_scale_mode = (
            AttentionScaleMode.F32_PER_TENSOR
            if _is_fp8_format(v_format)
            else AttentionScaleMode.E8M0_PER_1X32
        )
        return (
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            v_scale_mode,
        )
    if q_format in _MX_FORMATS:
        v_scale_mode = (
            AttentionScaleMode.F32_PER_CHANNEL
            if _is_fp8_format(v_format)
            else AttentionScaleMode.E8M0_PER_1X32
        )
        return (
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            v_scale_mode,
        )
    raise NotImplementedError(
        f"raw preprocessing is not implemented for Q/K format {q_format.name}"
    )


def _packed_lut_triple(
    kv_block_indices: Optional[Tensor],  # noqa: UP045
    lut_start: Optional[Tensor],  # noqa: UP045
    lut_count: Optional[Tensor],  # noqa: UP045
) -> Optional[tuple[Tensor, Tensor, Tensor]]:  # noqa: UP045
    present = (
        kv_block_indices is not None,
        lut_start is not None,
        lut_count is not None,
    )
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "kv_block_indices, lut_start, and lut_count must all be set or all omitted"
        )
    return kv_block_indices, lut_start, lut_count


def _block_mask_to_lut(
    block_mask: Tensor, query: Tensor, key: Tensor, block_tile: tuple[int, int]
) -> tuple[Tensor, Tensor, Tensor]:
    batch, query_length, query_heads, _ = query.shape
    key_length = key.shape[1]
    q_tile, kv_tile = block_tile
    q_tiles = (query_length + q_tile - 1) // q_tile
    kv_tiles = (key_length + kv_tile - 1) // kv_tile
    # The conversion below counts selected blocks with an arithmetic sum but fills the index list
    # from truthiness, so anything other than bool can make lut_count disagree with the entries
    # actually written.
    if block_mask.dtype != torch.bool:
        raise ValueError(f"block_mask must be a bool tensor, got {block_mask.dtype}")
    if block_mask.device != query.device:
        raise ValueError(
            f"block_mask must be on the same device as Q; got {block_mask.device} "
            f"and {query.device}"
        )
    if block_mask.dim() == 4:
        expected = (batch, query_heads, q_tiles, kv_tiles)
        if tuple(block_mask.shape) != expected:
            raise ValueError(
                "block_mask must have shape [batch, heads, "
                f"ceil(Sq/{q_tile}), ceil(Sk/{kv_tile})]; "
                f"got {tuple(block_mask.shape)}, expected {expected}"
            )
    elif block_mask.dim() == 3:
        expected = (batch, q_tiles, kv_tiles)
        if tuple(block_mask.shape) != expected:
            raise ValueError(
                "block_mask must have shape [batch, "
                f"ceil(Sq/{q_tile}), ceil(Sk/{kv_tile})] "
                f"or the 4-D per-head form; got {tuple(block_mask.shape)}, "
                f"expected {expected}"
            )
    else:
        raise ValueError(
            "block_mask must be 3-D [batch, Qtiles, KVtiles] or "
            "4-D [batch, heads, Qtiles, KVtiles]"
        )
    lut = block_attn_mask_to_ragged_lut(
        block_mask,
        num_heads=query_heads,
        return_none_if_dense=False,
    )
    if lut is None:
        raise RuntimeError("block_attn_mask_to_ragged_lut returned None")
    return lut


def _fmha_v4_fwd_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del out


@compile_ops(
    "module_fmha_v4_fwd",
    fc_name="fmha_v4_fwd",
    gen_fake=_fmha_v4_fwd_fake,
)
def _fmha_v4_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None: ...


@torch.library.custom_op("aiter::mha_v4_fwd_launch", mutates_args=("out",))
def _mha_v4_fwd_launch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    _fmha_v4_fwd(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale,
    )


@_mha_v4_fwd_launch.register_fake
def _mha_v4_fwd_launch_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale, out
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale


def _fmha_v4_fwd_sparse_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del kv_block_indices, lut_start, lut_count, q_tile, kv_tile
    del out


@compile_ops(
    "module_fmha_v4_fwd",
    fc_name="fmha_v4_fwd_sparse",
    gen_fake=_fmha_v4_fwd_sparse_fake,
)
def _fmha_v4_fwd_sparse(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None: ...


@torch.library.custom_op("aiter::mha_v4_fwd_sparse_launch", mutates_args=("out",))
def _mha_v4_fwd_sparse_launch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None:
    _fmha_v4_fwd_sparse(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale,
        kv_block_indices,
        lut_start,
        lut_count,
        q_tile,
        kv_tile,
    )


@_mha_v4_fwd_sparse_launch.register_fake
def _mha_v4_fwd_sparse_launch_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale, out
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del kv_block_indices, lut_start, lut_count, q_tile, kv_tile


def _fmha_v4_fwd_sol_attn_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    mean_k: Tensor,
    mean_v: Tensor,
    block_bitmap: Tensor,
    mean_k_scale: Optional[Tensor] = None,  # noqa: UP045
    mean_v_scale: Optional[Tensor] = None,  # noqa: UP045
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del kv_block_indices, lut_start, lut_count, q_tile, kv_tile
    del mean_k, mean_v, block_bitmap, mean_k_scale, mean_v_scale
    del out


@compile_ops(
    "module_fmha_v4_fwd",
    fc_name="fmha_v4_fwd_sol_attn",
    gen_fake=_fmha_v4_fwd_sol_attn_fake,
)
def _fmha_v4_fwd_sol_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    mean_k: Tensor,
    mean_v: Tensor,
    block_bitmap: Tensor,
    mean_k_scale: Optional[Tensor] = None,  # noqa: UP045
    mean_v_scale: Optional[Tensor] = None,  # noqa: UP045
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None: ...


@torch.library.custom_op("aiter::mha_v4_fwd_sol_attn_launch", mutates_args=("out",))
def _mha_v4_fwd_sol_attn_launch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    mean_k: Tensor,
    mean_v: Tensor,
    block_bitmap: Tensor,
    mean_k_scale: Optional[Tensor] = None,  # noqa: UP045
    mean_v_scale: Optional[Tensor] = None,  # noqa: UP045
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None:
    _fmha_v4_fwd_sol_attn(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale,
        kv_block_indices,
        lut_start,
        lut_count,
        mean_k,
        mean_v,
        block_bitmap,
        mean_k_scale,
        mean_v_scale,
        q_tile,
        kv_tile,
    )


@_mha_v4_fwd_sol_attn_launch.register_fake
def _mha_v4_fwd_sol_attn_launch_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
    mean_k: Tensor,
    mean_v: Tensor,
    block_bitmap: Tensor,
    mean_k_scale: Optional[Tensor] = None,  # noqa: UP045
    mean_v_scale: Optional[Tensor] = None,  # noqa: UP045
    q_tile: int = 0,
    kv_tile: int = 0,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale, out
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del kv_block_indices, lut_start, lut_count, q_tile, kv_tile
    del mean_k, mean_v, block_bitmap, mean_k_scale, mean_v_scale


def _sol_attn_triple(
    mean_k: Optional[Tensor],  # noqa: UP045
    mean_v: Optional[Tensor],  # noqa: UP045
    block_bitmap: Optional[Tensor],  # noqa: UP045
) -> Optional[tuple[Tensor, Tensor, Tensor]]:  # noqa: UP045
    present = (
        mean_k is not None,
        mean_v is not None,
        block_bitmap is not None,
    )
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "mean_k, mean_v, and block_bitmap must all be set or all omitted"
        )
    return mean_k, mean_v, block_bitmap


def mha_v4_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    q_scale_mode: AttentionScaleMode,
    k_scale_mode: AttentionScaleMode,
    v_scale_mode: AttentionScaleMode,
    softmax_scale: Optional[float] = None,  # noqa: UP045
    out: Optional[Tensor] = None,  # noqa: UP045
    return_lse: bool = False,
    kv_block_indices: Optional[Tensor] = None,  # noqa: UP045
    lut_start: Optional[Tensor] = None,  # noqa: UP045
    lut_count: Optional[Tensor] = None,  # noqa: UP045
    mean_k: Optional[Tensor] = None,  # noqa: UP045
    mean_v: Optional[Tensor] = None,  # noqa: UP045
    block_bitmap: Optional[Tensor] = None,  # noqa: UP045
    mean_k_scale: Optional[Tensor] = None,  # noqa: UP045
    mean_v_scale: Optional[Tensor] = None,  # noqa: UP045
    block_tile: Optional[tuple[int, int]] = None,  # noqa: UP045
) -> Tensor:
    """Launch non-causal MHA v4 over pre-quantized BSHD operands.

    Formats and scale modes select an explicit ASM row. Packed widths and
    nonstandard K layouts are validated before launch; output is BF16 BSHD.
    Pass the ragged LUT triple to select the sorted-sparse row; omit all three
    tensors for dense. The work table is built inside the sparse custom op.
    Adding the pooled triple (mean_k, mean_v, block_bitmap) on top of the LUT
    selects Sol-Attn instead, which corrects the blocks the LUT dropped rather
    than discarding them; aiter.ops.triton's sol_attn_prepare() builds all six
    tensors from one mask. A row that selects nothing falls back to the
    pooled-only softmax over every block instead of to a zero tile.
    A block-granular operand also needs its pooled scale (mean_k_scale,
    mean_v_scale), which sol_attn_prepare() returns for exactly the operands
    whose source scale could not survive pooling.

    block_tile names the (q_tile, kv_tile) geometry the LUT and pooled tensors
    were built for; it defaults to mha_v4_block_tile(). Pass 64x64 on gfx950 to
    route FP8 at a 64-token block instead of 128, which costs throughput per
    token and only pays off if the finer blocks drop enough attention mass --
    mha_v4_block_tiles() lists what the GPU has kernels for.
    """
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    lut = _packed_lut_triple(kv_block_indices, lut_start, lut_count)
    pooled = _sol_attn_triple(mean_k, mean_v, block_bitmap)
    if pooled is not None and lut is None:
        raise ValueError(
            "Sol-Attn MHA v4 needs the ragged LUT triple as well: the pooled pass corrects the "
            "blocks the LUT did not compute exactly, so it has no meaning without one"
        )
    if pooled is None and (mean_k_scale is not None or mean_v_scale is not None):
        raise ValueError(
            "a pooled scale only describes a pooled operand: pass mean_k / mean_v / block_bitmap "
            "alongside it"
        )
    expected_scale_modes = scale_modes_for_formats(q_format, k_format, v_format)
    scale_modes = (q_scale_mode, k_scale_mode, v_scale_mode)
    mxfp8_scale_modes = (
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.F32_PER_TENSOR,
    )
    is_mxfp8_recipe = (
        q_format in _FP8_FORMATS
        and k_format == q_format
        and v_format == q_format
        and scale_modes == mxfp8_scale_modes
    )
    if scale_modes != expected_scale_modes and not is_mxfp8_recipe:
        raise ValueError(
            "unsupported scale recipe for formats: "
            f"got {tuple(mode.name for mode in scale_modes)}, "
            f"expected {tuple(mode.name for mode in expected_scale_modes)}"
        )

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("MHA v4 expects BSHD Q, K, and V tensors")
    batch, query_length, query_heads, _ = q.shape
    if k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError("Q, K, and V must have the same batch size")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError("K and V must have matching sequence and head dimensions")
    kv_heads = k.shape[2]
    if kv_heads == 0:
        raise ValueError("MHA v4 requires non-empty KV heads")
    if query_heads % kv_heads != 0:
        raise ValueError("MHA v4 requires query heads to be divisible by KV heads")
    gqa_ratio = query_heads // kv_heads
    if gqa_ratio > 16 or gqa_ratio & (gqa_ratio - 1):
        raise ValueError("MHA v4 supports power-of-two GQA ratios up to 16")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("MHA v4 expects GPU tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Q, K, and V must have contiguous last dimensions")

    logical_head_dim = 128
    expected_q_width = _PACKED_QK_WIDTH[q_format]
    if q.shape[-1] != expected_q_width or k.shape[-1] != expected_q_width:
        raise ValueError(
            f"{q_format.name} Q/K must have packed width {expected_q_width}"
        )
    if v.shape[-1] != logical_head_dim:
        raise ValueError("MHA v4 currently requires logical V head dimension 128")
    if q_format == AttentionFormat.MXFP4:
        tiles = (k.shape[1] + 127) // 128
        head_stride = tiles * 8192
        expected_k_stride = (k.shape[2] * head_stride, 64, head_stride, 1)
        if k.stride() != expected_k_stride:
            raise ValueError("MXFP4 K must use the coalesced MHA v4 tile layout")

    if softmax_scale is None:
        softmax_scale = logical_head_dim**-0.5
    if out is None:
        out = torch.empty(
            (batch, query_length, query_heads, logical_head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    elif out.shape != (batch, query_length, query_heads, logical_head_dim):
        raise ValueError("out has the wrong shape for MHA v4")
    elif out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must be a BF16 tensor on the same device as Q")

    launch_args = (
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        int(q_format),
        int(k_format),
        int(v_format),
        int(q_scale_mode),
        int(k_scale_mode),
        int(v_scale_mode),
        softmax_scale,
    )
    if lut is None:
        _mha_v4_fwd_launch(*launch_args)
    else:
        mode_name = "Sol-Attn" if pooled is not None else "sorted-sparse"
        q_tile, kv_tile = (
            mha_v4_block_tile() if block_tile is None else tuple(block_tile)
        )
        # Scalar query, not a membership test against mha_v4_block_tiles(): this sits on the
        # torch.compile path, where only the guarded per-Q-tile read is traceable. Asked for these
        # operands rather than for any, because a geometry can be precision-specific and the
        # alternative is the launch failing on a missing row naming format codes and no tile.
        operands = mha_v4_operands(
            q_format, k_format, v_format, q_scale_mode, k_scale_mode, v_scale_mode
        )
        mode = MHA_V4_SOL_ATTN_MODE if pooled is not None else MHA_V4_SPARSE_MODE
        if mha_v4_kv_tile_for_q_tile(q_tile, operands, mode) != kv_tile:
            raise ValueError(
                f"{mode_name} MHA v4 has no {q_tile}x{kv_tile} kernel on this GPU for "
                f"q={q_format.name} k={k_format.name} v={v_format.name}; those operands have "
                f"{mha_v4_block_tiles(operands, mode)} and the GPU has "
                f"{mha_v4_block_tiles(mode=mode)} in some precision"
            )
        if k.shape[1] % kv_tile != 0:
            raise ValueError(
                f"{mode_name} MHA v4 requires key length padded to a "
                f"multiple of {kv_tile}"
            )
        if pooled is None:
            _mha_v4_fwd_sparse_launch(*launch_args, *lut, q_tile, kv_tile)
        else:
            _mha_v4_fwd_sol_attn_launch(
                *launch_args,
                *lut,
                *pooled,
                mean_k_scale,
                mean_v_scale,
                q_tile,
                kv_tile,
            )
    return out


def _quantize_per_tensor(
    input: Tensor, output_dtype: torch.dtype, dtype_max: float, clip: float
) -> tuple[Tensor, Tensor]:
    if not input.is_contiguous():
        raise ValueError("MHA v4 per-tensor quantization requires contiguous input")
    numel = input.numel()
    blocks = triton.cdiv(numel, MHA_V4_PER_TENSOR_BLOCK_SIZE)
    partial = input.new_empty((blocks,), dtype=torch.float32)
    scale = input.new_empty((1,), dtype=torch.float32)
    output = input.new_empty(input.shape, dtype=output_dtype)
    mha_v4_per_tensor_amax_kernel[(blocks,)](
        input,
        partial,
        numel,
        BLOCK_SIZE=MHA_V4_PER_TENSOR_BLOCK_SIZE,
        num_warps=8,
    )
    scale_block = triton.next_power_of_2(blocks)
    mha_v4_per_tensor_scale_kernel[(1,)](
        partial,
        scale,
        blocks,
        dtype_max=dtype_max / clip,
        BLOCK_SIZE=scale_block,
        num_warps=8,
    )
    mha_v4_per_tensor_quant_kernel[(blocks,)](
        input,
        output,
        scale,
        numel,
        IS_INT8=output_dtype == torch.int8,
        BLOCK_SIZE=MHA_V4_PER_TENSOR_BLOCK_SIZE,
        num_warps=8,
    )
    return output, scale


@torch.library.custom_op("aiter::mha_v4_quantize_int8_v2", mutates_args=())
def quantize_int8(input: Tensor, clip: float = 1.0) -> tuple[Tensor, Tensor]:
    """Per-tensor quantize a contiguous tensor to INT8 and return its scale."""
    return _quantize_per_tensor(input, torch.int8, 127.0, clip)


@quantize_int8.register_fake
def _quantize_int8_fake(input: Tensor, clip: float = 1.0) -> tuple[Tensor, Tensor]:
    del clip
    return input.new_empty(input.shape, dtype=torch.int8), input.new_empty(
        (1,), dtype=torch.float32
    )


@torch.library.custom_op("aiter::mha_v4_quantize_fp8", mutates_args=())
def quantize_fp8(input: Tensor) -> tuple[Tensor, Tensor]:
    """Per-tensor quantize a contiguous tensor to native FP8 and return its scale."""
    return _quantize_per_tensor(input, dtypes.fp8, torch.finfo(dtypes.fp8).max, 1.0)


@quantize_fp8.register_fake
def _quantize_fp8_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (1,), dtype=torch.float32
    )


def quantize_fp8_rotated(input: Tensor) -> tuple[Tensor, Tensor]:
    """Apply normalized hd128 Walsh-Hadamard rotation, then per-tensor FP8 quantize."""
    if input.shape[-1] != 128 or not input.is_contiguous():
        raise ValueError("rotated FP8 quantization requires contiguous hd128 input")
    rotated = torch.empty_like(input)
    rotate_activation_hd128(rotated, input)
    return quantize_fp8(rotated)


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp8_q", mutates_args=())
def quantize_mxfp8_q(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    """Rotate and quantize hd128 BSHD Q to MXFP8 data and E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(input, "MXFP8 quantization")
    quantized = input.new_empty(input.shape, dtype=dtypes.fp8)
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp8_quant(quantized, scale, input, multiplier)
    return quantized, scale


@quantize_mxfp8_q.register_fake
def _quantize_mxfp8_q_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (batch, sequence, heads, head_dim // 32), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp8_k", mutates_args=())
def quantize_mxfp8_k(input: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate and quantize hd128 BSHD K to MXFP8 data and E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(
        input, "MXFP8 K quantization"
    )
    quantized = input.new_empty(input.shape, dtype=dtypes.fp8)
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp8_quant(quantized, scale, input, 1.0)
    return quantized, scale


@quantize_mxfp8_k.register_fake
def _quantize_mxfp8_k_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (batch, sequence, heads, head_dim // 32), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp4", mutates_args=())
def quantize_mxfp4_q(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD Q as MXFP4 data with E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(input, "MXFP4 quantization")
    quantized = input.new_empty(
        (batch, sequence, heads, head_dim // 2), dtype=torch.uint8
    )
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp4_quant(quantized, scale, input, multiplier)
    return quantized, scale


@quantize_mxfp4_q.register_fake
def _quantize_mxfp4_q_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (batch, sequence, heads, head_dim // 2), dtype=torch.uint8
    ), input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)


def mxfp4_k_raw_buffer_size(batch: int, sequence: int, heads: int) -> int:
    """Return bytes for the coalesced MXFP4 K backing buffer."""
    tiles = (sequence + 127) // 128
    return batch * heads * tiles * 8192


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp4_k_raw", mutates_args=())
def quantize_mxfp4_k(input: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD K into the coalesced MXFP4 ASM layout."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(
        input, "MXFP4 K quantization"
    )
    raw = input.new_empty(
        (mxfp4_k_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    )
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp4_quant_k(raw, scale, input)
    return raw, scale


@quantize_mxfp4_k.register_fake
def _quantize_mxfp4_k_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (mxfp4_k_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    ), input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)


def mxfp4_k_view(raw: Tensor, scale: Tensor) -> Tensor:
    """Rebuild the logical MXFP4 K view from its contiguous backing buffer."""
    batch, sequence, heads, _ = scale.shape
    tiles = (sequence + 127) // 128
    head_stride = tiles * 8192
    return torch.as_strided(
        raw,
        (batch, sequence, heads, 64),
        (heads * head_stride, 64, head_stride, 1),
    )


def mxfp6_k_view(
    raw: Tensor,
    scale_raw: Tensor,
    batch: int,
    sequence: int,
    heads: int,
) -> tuple[Tensor, Tensor]:
    """Rebuild the logical MXFP6 K and scale views from raw backing buffers."""
    return fp6_k_lds_order_views_from_raw(raw, scale_raw, batch, sequence, heads)


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp6_q", mutates_args=())
def quantize_mxfp6_q(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD Q as MXFP6 E2M3 with E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(
        input, "MXFP6 E2M3 Q quantization"
    )
    quantized = input.new_empty(
        (batch, sequence, heads, head_dim // 32 * 24), dtype=torch.uint8
    )
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp6_quant(quantized, scale, input, multiplier)
    return quantized, scale


@quantize_mxfp6_q.register_fake
def _quantize_mxfp6_q_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (batch, sequence, heads, head_dim // 32 * 24), dtype=torch.uint8
    ), input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp6_k_raw", mutates_args=())
def quantize_mxfp6_k(input: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD K into raw MXFP6 ASM data and scale buffers."""
    batch, sequence, heads, _ = _validate_bshd_hd128(input, "MXFP6 E2M3 K quantization")
    data_size, scale_size = fp6_k_raw_buffer_sizes(batch, sequence, heads)
    raw = input.new_empty((data_size,), dtype=torch.uint8)
    scale_raw = input.new_empty((scale_size,), dtype=torch.uint8)
    rotate_activation_mxfp6_quant_k(raw, scale_raw, input)
    return raw, scale_raw


@quantize_mxfp6_k.register_fake
def _quantize_mxfp6_k_raw_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, _ = input.shape
    data_size, scale_size = fp6_k_raw_buffer_sizes(batch, sequence, heads)
    return input.new_empty((data_size,), dtype=torch.uint8), input.new_empty(
        (scale_size,), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_fp8", mutates_args=())
def quantize_v_fp8(input: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize hd128 BSHD V to FP8 with one FP32 scale per channel."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(input, "FP8 V quantization")
    fp8_max = torch.finfo(dtypes.fp8).max
    scale_block_k = 256
    scale_blocks = triton.cdiv(sequence, scale_block_k)
    scale_reduce_block = triton.next_power_of_2(scale_blocks)
    partial = input.new_empty(
        (batch * heads, scale_blocks, head_dim), dtype=torch.float32
    )
    scale = input.new_empty((batch, heads, head_dim), dtype=torch.float32)
    sage_quant_v_amax_partial_kernel[(batch * heads * scale_blocks,)](
        input,
        partial,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        sequence,
        heads,
        scale_blocks,
        D=head_dim,
        BLOCK_K=scale_block_k,
        num_warps=8,
    )
    sage_quant_v_amax_finalize_kernel[(triton.cdiv(head_dim, 32), batch * heads)](
        partial,
        scale,
        scale_blocks,
        D=head_dim,
        FP8_MAX=fp8_max,
        BLOCK_N=scale_reduce_block,
        BLOCK_D=32,
        num_warps=4,
    )
    block_k = 64
    blocks = triton.cdiv(sequence, block_k)
    quantized = torch.empty_like(input, dtype=dtypes.fp8)
    sage_quant_v_kernel[(batch * heads * blocks,)](
        input,
        quantized,
        scale,
        input.stride(0),
        input.stride(2),
        input.stride(1),
        input.stride(3),
        scale.stride(0),
        scale.stride(1),
        batch,
        heads,
        blocks,
        sequence,
        D=head_dim,
        BLK_K=block_k,
        num_stages=3,
        num_warps=8,
    )
    return quantized, scale


@quantize_v_fp8.register_fake
def _quantize_v_fp8_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, _, heads, head_dim = input.shape
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (batch, heads, head_dim), dtype=torch.float32
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_mxfp4_raw_v2", mutates_args=())
def quantize_v_mxfp4(input: Tensor) -> tuple[Tensor, Tensor]:
    """Pack hd128 BSHD V into raw column-major MXFP4 data and scale buffers."""
    _validate_bshd_hd128(input, "MXFP4 V quantization")
    return pack_v_mxfp4_colmajor_raw(input)


@quantize_v_mxfp4.register_fake
def _quantize_v_mxfp4_raw_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, _ = input.shape
    tiles = fp4_v_padded_sequence(sequence) // 128
    return input.new_empty(
        (fp4_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    ), input.new_empty((batch, heads, tiles * 512), dtype=torch.uint8)


@torch.library.custom_op("aiter::mha_v4_quantize_v_mxfp6", mutates_args=())
def quantize_v_mxfp6(input: Tensor) -> tuple[Tensor, Tensor]:
    """Pack hd128 BSHD V into MXFP6 data and E8M0 scale views."""
    _validate_bshd_hd128(input, "MXFP6 V quantization")
    return pack_fp6_v_data_scale_views(input)


@quantize_v_mxfp6.register_fake
def _quantize_v_mxfp6_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    tiles = (sequence + 127) // 128
    head_stride = tiles * 12288
    raw = input.new_empty((batch * heads * head_stride + 256,), dtype=torch.uint8)
    quantized = torch.as_strided(
        raw,
        (batch, sequence, heads, head_dim),
        (heads * head_stride, 96, head_stride, 1),
    )
    return quantized, input.new_empty((batch, heads, tiles * 512), dtype=torch.uint8)


def mxfp4_v_view(raw: Tensor, scale: Tensor, sequence: int) -> Tensor:
    """Rebuild the logical MXFP4 V view from its contiguous backing buffer."""
    batch, heads, _ = scale.shape
    padded_sequence = fp4_v_padded_sequence(sequence)
    return torch.as_strided(
        raw,
        (batch, sequence, heads, 128),
        (heads * padded_sequence * 64, 64, padded_sequence * 64, 1),
    )


@torch.library.custom_op(
    "aiter::mha_v4_launch_mxfp4_coalesced_v2", mutates_args=("out",)
)
def _launch_mxfp4_coalesced(
    q: Tensor,
    q_descale: Tensor,
    k_data: Tensor,
    k_descale: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    v_format: int,
    softmax_scale: float,
) -> None:
    resolved_v_format = AttentionFormat(v_format)
    k = mxfp4_k_view(k_data, k_descale)
    v = (
        v_data
        if _is_fp8_format(resolved_v_format)
        else mxfp4_v_view(v_data, v_descale, k.shape[1])
    )
    scale_modes = scale_modes_for_formats(
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        resolved_v_format,
    )
    mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        resolved_v_format,
        *scale_modes,
        softmax_scale=softmax_scale,
        out=out,
    )


@_launch_mxfp4_coalesced.register_fake
def _launch_mxfp4_coalesced_fake(
    q: Tensor,
    q_descale: Tensor,
    k_data: Tensor,
    k_descale: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    v_format: int,
    softmax_scale: float,
) -> None:
    del q, q_descale, k_data, k_descale, v_data, v_descale, v_format, softmax_scale
    del out


@torch.library.custom_op("aiter::mha_v4_launch_mxfp6_v2", mutates_args=("out",))
def _launch_mxfp6(
    q: Tensor,
    q_descale: Tensor,
    k_raw: Tensor,
    k_descale_raw: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    sequence_k: int,
    heads: int,
    v_format: int,
    softmax_scale: float,
) -> None:
    resolved_v_format = AttentionFormat(v_format)
    k, k_descale = mxfp6_k_view(k_raw, k_descale_raw, q.shape[0], sequence_k, heads)
    v = (
        v_data
        if _is_fp8_format(resolved_v_format)
        else mxfp4_v_view(v_data, v_descale, sequence_k)
    )
    scale_modes = scale_modes_for_formats(
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        resolved_v_format,
    )
    mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        resolved_v_format,
        *scale_modes,
        softmax_scale=softmax_scale,
        out=out,
    )


@_launch_mxfp6.register_fake
def _launch_mxfp6_fake(
    q: Tensor,
    q_descale: Tensor,
    k_raw: Tensor,
    k_descale_raw: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    sequence_k: int,
    heads: int,
    v_format: int,
    softmax_scale: float,
) -> None:
    del q, q_descale, k_raw, k_descale_raw, v_data, v_descale
    del sequence_k, heads, v_format, softmax_scale
    del out


def _validate_mha_v4_raw_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Optional[Tensor],  # noqa: UP045
    operation: str,
) -> Tensor:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"{operation} expects BSHD Q, K, and V tensors")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError(f"{operation} currently expects BF16 Q, K, and V inputs")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError(f"{operation} currently supports head dimension 128 only")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError(f"{operation} currently requires contiguous BSHD inputs")
    if out is None:
        return torch.empty_like(q, dtype=torch.bfloat16)
    if out.shape != q.shape or out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must match Q's shape/device and have BF16 dtype")
    return out


def mha_v4_mxfp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,  # noqa: UP045
    out: Optional[Tensor] = None,  # noqa: UP045
    return_lse: bool = False,
    block_mask: Optional[Tensor] = None,  # noqa: UP045
) -> Tensor:
    """Quantize BF16 BSHD Q/K to MXFP8 and V to per-tensor FP8.

    K and V may have fewer heads than Q for GQA. The Q-to-KV head ratio must
    be a power of two no greater than 16; output retains Q's head count.
    Optional ``block_mask`` selects the sorted-sparse row (gfx950 MXFP8 only);
    LUT rows are one per query head, and K/V addressing uses the GQA ratio.
    """
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    out = _validate_mha_v4_raw_inputs(q, k, v, out, "mha_v4_mxfp8")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lut_indices: Optional[Tensor] = None  # noqa: UP045
    lut_start: Optional[Tensor] = None  # noqa: UP045
    lut_count: Optional[Tensor] = None  # noqa: UP045
    if block_mask is not None:
        lut_indices, lut_start, lut_count = _block_mask_to_lut(
            block_mask, q, k, mha_v4_block_tile()
        )

    fp8_format = native_fp8_format()
    q_quantized, q_descale = quantize_mxfp8_q(q, mha_v4_q_multiplier(softmax_scale))
    k_quantized, k_descale = quantize_mxfp8_k(k)
    v_quantized, v_descale = quantize_fp8(v)
    return mha_v4_packed(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=softmax_scale,
        out=out,
        kv_block_indices=lut_indices,
        lut_start=lut_start,
        lut_count=lut_count,
    )


def mha_v4(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    softmax_scale: Optional[float] = None,  # noqa: UP045
    out: Optional[Tensor] = None,  # noqa: UP045
    return_lse: bool = False,
    block_mask: Optional[Tensor] = None,  # noqa: UP045
    block_tile: Optional[tuple[int, int]] = None,  # noqa: UP045
) -> Tensor:
    """Quantize BF16 BSHD operands and run non-causal MHA v4.

    Q and K formats must match. The selected Q/K/V recipe determines canonical
    quantizers, scale modes, and the packed ASM row; output is BF16 BSHD.
    K and V may have fewer heads than Q for GQA. The Q-to-KV head ratio must
    be a power of two no greater than 16; output retains Q's head count.
    ``block_mask`` is optional boolean tile metadata: ``[B, H, Qtiles, KVtiles]``
    or ``[B, Qtiles, KVtiles]`` (broadcast heads). Its geometry is ``block_tile``,
    defaulting to mha_v4_block_tile() -- 256x128 on gfx950, 256x64 on gfx942 --
    and gfx950 FP8 also accepts 64x64 for finer routing. Sparse LUT rows are one
    per query head; K/V addressing uses the GQA ratio. A row may select nothing:
    an all-False row is a no-op that writes a zero output tile.
    """
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    out = _validate_mha_v4_raw_inputs(q, k, v, out, "mha_v4")
    q_scale_mode, k_scale_mode, v_scale_mode = scale_modes_for_formats(
        q_format, k_format, v_format
    )

    lut_indices: Optional[Tensor] = None  # noqa: UP045
    lut_start: Optional[Tensor] = None  # noqa: UP045
    lut_count: Optional[Tensor] = None  # noqa: UP045
    if block_mask is not None:
        lut_indices, lut_start, lut_count = _block_mask_to_lut(
            block_mask, q, k, mha_v4_block_tile() if block_tile is None else block_tile
        )
    packed_lut = {
        "kv_block_indices": lut_indices,
        "lut_start": lut_start,
        "lut_count": lut_count,
        "block_tile": block_tile,
    }
    if q_format == AttentionFormat.BF16:
        # Q and K pass through unquantized, so their descale arguments are placeholders that the
        # NONE scale mode makes the kernel ignore. V is the only operand that can be quantized
        # here, and only to per-tensor FP8.
        v_quantized, v_descale = (
            quantize_fp8(v) if _is_fp8_format(v_format) else (v, v)
        )
        return mha_v4_packed(
            q,
            k,
            v_quantized,
            q,
            k,
            v_descale,
            q_format,
            k_format,
            v_format,
            q_scale_mode,
            k_scale_mode,
            v_scale_mode,
            softmax_scale=softmax_scale,
            out=out,
            return_lse=return_lse,
            **packed_lut,
        )
    if q_format == AttentionFormat.INT8 and _is_fp8_format(v_format):
        q_quantized, q_descale = quantize_int8(q)
        k_quantized, k_descale = quantize_int8(k)
        v_quantized, v_descale = quantize_fp8(v)
    elif q_format in _FP8_FORMATS and v_format in (
        q_format,
        AttentionFormat.MXFP6,
    ):
        q_quantized, q_descale = quantize_fp8_rotated(q)
        k_quantized, k_descale = quantize_fp8_rotated(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = quantize_fp8(v)
        else:
            v_quantized, v_descale = quantize_v_mxfp6(v)
    elif q_format == AttentionFormat.MXFP4 and v_format in (
        *_FP8_FORMATS,
        AttentionFormat.MXFP4,
    ):
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        q_quantized, q_descale = quantize_mxfp4_q(q, mha_v4_q_multiplier(softmax_scale))
        k_quantized, k_descale = quantize_mxfp4_k(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = quantize_v_fp8(v)
        else:
            v_quantized, v_descale = quantize_v_mxfp4(v)
        if lut_indices is None:
            _launch_mxfp4_coalesced(
                q_quantized,
                q_descale,
                k_quantized,
                k_descale,
                v_quantized,
                v_descale,
                out,
                int(v_format),
                softmax_scale,
            )
            return out
        k_view = mxfp4_k_view(k_quantized, k_descale)
        v_view = (
            v_quantized
            if _is_fp8_format(v_format)
            else mxfp4_v_view(v_quantized, v_descale, k.shape[1])
        )
        return mha_v4_packed(
            q_quantized,
            k_view,
            v_view,
            q_descale,
            k_descale,
            v_descale,
            q_format,
            k_format,
            v_format,
            q_scale_mode,
            k_scale_mode,
            v_scale_mode,
            softmax_scale=softmax_scale,
            out=out,
            return_lse=return_lse,
            **packed_lut,
        )
    elif q_format == AttentionFormat.MXFP6 and v_format in (
        *_FP8_FORMATS,
        AttentionFormat.MXFP4,
    ):
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        q_quantized, q_descale = quantize_mxfp6_q(q, mha_v4_q_multiplier(softmax_scale))
        k_quantized, k_descale = quantize_mxfp6_k(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = quantize_v_fp8(v)
        else:
            v_quantized, v_descale = quantize_v_mxfp4(v)
        if lut_indices is None:
            _launch_mxfp6(
                q_quantized,
                q_descale,
                k_quantized,
                k_descale,
                v_quantized,
                v_descale,
                out,
                k.shape[1],
                k.shape[2],
                int(v_format),
                softmax_scale,
            )
            return out
        k_view, k_descale_view = mxfp6_k_view(
            k_quantized, k_descale, q.shape[0], k.shape[1], k.shape[2]
        )
        v_view = (
            v_quantized
            if _is_fp8_format(v_format)
            else mxfp4_v_view(v_quantized, v_descale, k.shape[1])
        )
        return mha_v4_packed(
            q_quantized,
            k_view,
            v_view,
            q_descale,
            k_descale_view,
            v_descale,
            q_format,
            k_format,
            v_format,
            q_scale_mode,
            k_scale_mode,
            v_scale_mode,
            softmax_scale=softmax_scale,
            out=out,
            return_lse=return_lse,
            **packed_lut,
        )
    else:
        raise NotImplementedError(
            "raw preprocessing is not implemented yet for "
            f"Q={q_format.name}, K={k_format.name}, V={v_format.name}"
        )

    return mha_v4_packed(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale=softmax_scale,
        out=out,
        return_lse=return_lse,
        **packed_lut,
    )


def mha_v4_sol_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    beta: float = 0.4,
    softmax_scale: Optional[float] = None,  # noqa: UP045
    out: Optional[Tensor] = None,  # noqa: UP045
    return_lse: bool = False,
    block_tile: Optional[tuple[int, int]] = None,  # noqa: UP045
) -> Tensor:
    """Quantize BF16 BSHD operands, route the blocks, and run non-causal Sol-Attn MHA v4.

    Sol-Attn (arXiv 2607.24027) computes the above-threshold KV blocks exactly and recovers the
    rest from pooled per-block K/V under the same online softmax, so dropping a block costs its
    higher-order terms rather than all of its mass. ``beta`` sets the threshold at
    ``mean_j(proxy) + beta * std_j(proxy)`` per query tile, so it selects a block *density* rather
    than a block count: larger beta keeps fewer blocks exact and leans harder on the correction.

    Unlike ``mha_v4(block_mask=...)`` the selection is not the caller's to pass, because routing
    has to see the quantized K that the kernel will read. Everything here is traceable, so this
    composes under torch.compile(fullgraph=True).

    Only recipes with a mode-2 manifest row are supported: the pooled K/V are pooled in the source
    dtype and reuse the source descales, which holds for per-tensor scales and for the BF16 rows
    that have no descale at all, but not for the block-granular MX ones, which would need pooled
    scales of their own.

    ``block_tile`` sets the routing and dispatch geometry, defaulting to mha_v4_block_tile(). A
    finer tile raises beta's resolution -- the threshold is per query tile, so 64x64 gives a block
    a quarter as much mass to hide behind -- at the cost of more work per token; gfx950 FP8 is the
    only recipe with a 64x64 row. Routing, pooling and the kernel all read it from here, so they
    cannot end up disagreeing.
    """
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    is_fp8_recipe = (
        q_format in _FP8_FORMATS and k_format == q_format and v_format == q_format
    )
    is_i8fp8_recipe = (
        q_format == AttentionFormat.INT8
        and k_format == q_format
        and _is_fp8_format(v_format)
    )
    is_bf16_qk_recipe = q_format == AttentionFormat.BF16 and k_format == q_format
    if not (is_fp8_recipe or is_i8fp8_recipe or is_bf16_qk_recipe):
        raise NotImplementedError(
            "Sol-Attn MHA v4 currently has manifest rows for the per-tensor FP8, i8fp8, bf16 "
            f"and bf16fp8 recipes only; got Q={q_format.name}, K={k_format.name}, "
            f"V={v_format.name}"
        )
    out = _validate_mha_v4_raw_inputs(q, k, v, out, "mha_v4_sol_attn")
    q_scale_mode, k_scale_mode, v_scale_mode = scale_modes_for_formats(
        q_format, k_format, v_format
    )

    if is_bf16_qk_recipe:
        # Nothing to quantize on the Q/K side, so the descales are placeholders the NONE scale
        # mode makes the kernel ignore, exactly as in mha_v4().
        q_quantized, q_descale = q, q
        k_quantized, k_descale = k, k
    elif is_i8fp8_recipe:
        q_quantized, q_descale = quantize_int8(q)
        k_quantized, k_descale = quantize_int8(k)
    else:
        q_quantized, q_descale = quantize_fp8_rotated(q)
        k_quantized, k_descale = quantize_fp8_rotated(k)
    v_quantized, v_descale = (
        quantize_fp8(v) if _is_fp8_format(v_format) else (v, v)
    )

    # Routed from the quantized K/V, not the BF16 inputs: the proxy scores have to be the ones the
    # kernel's exact pass will reproduce, or a block sitting within rounding distance of the
    # threshold can be selected here and skipped there.
    tile_m, tile_n = mha_v4_block_tile() if block_tile is None else block_tile
    plan = sol_attn_prepare(
        q_quantized,
        k_quantized,
        v_quantized,
        beta=beta,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        num_heads=q.shape[2],
    )

    return mha_v4_packed(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale=softmax_scale,
        out=out,
        return_lse=return_lse,
        kv_block_indices=plan["kv_block_indices"],
        lut_start=plan["lut_start"],
        lut_count=plan["lut_count"],
        mean_k=plan["mean_k"],
        mean_v=plan["mean_v"],
        block_bitmap=plan["block_bitmap"],
        block_tile=(tile_m, tile_n),
    )
