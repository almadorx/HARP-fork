"""Packed matrix multiplication for residual vector codebooks."""

import torch
import quiptools_cuda

CODEBOOK_BITS = {
    "lib.codebook.latticee8_padded12": 2,
    "lib.codebook.latticee8_padded12_rvq3bit": 3,
    "lib.codebook.latticee8_padded12_rvq4bit": 4,
}


def residual_lookup(inputs, indices, grid, scale):
    """Apply the one-bit residual codebook to up to eight rows."""
    rows, width = inputs.shape
    padded = inputs.new_zeros((8, width))
    padded[:rows].copy_(inputs)
    output = torch.empty(
        (8, indices.shape[0]), device=inputs.device, dtype=torch.float32
    )
    quiptools_cuda.lookupmatmul_e81b_k8(padded / scale, indices, grid, output)
    return output[:rows]


def decode_three(inputs, indices, codebook, residual_scale, decode_batch=None):
    """Apply the primary and one-bit residual codebooks."""
    rows, width = inputs.shape
    outputs = indices[1].shape[0]
    primary_indices = indices[0].view(outputs // 16, width // 64, 8, 4)
    if rows == 1 or decode_batch is not None:
        residual = residual_lookup(
            inputs, indices[1], codebook.e81b_grid, residual_scale
        )
        if rows == 1:
            primary = torch.ops.quip_lib.decode_matvec_e8p(
                inputs[0], primary_indices, codebook.grid_packed_abs, outputs, width
            ).unsqueeze(0)
        else:
            primary = decode_batch(inputs, primary_indices, codebook.grid_packed_abs)
        result = primary + residual
        return result if rows == 1 else result.half()
    primary = quiptools_cuda.decompress_packed_e8p(
        primary_indices, codebook.grid_packed_abs
    )
    residual = torch.empty((outputs, width), device=inputs.device, dtype=torch.float16)
    quiptools_cuda.decompress_e81b_packed(indices[1], codebook.e81b_grid, residual)
    return inputs @ (primary + residual / residual_scale).T


def decode_packed(inputs, indices, codebook, bits, residual_scale, decode_batch=None):
    """Multiply FP16 rows by a two-, three- or four-bit packed matrix."""
    if bits == 3:
        return decode_three(inputs, indices, codebook, residual_scale, decode_batch)
    rows, width = inputs.shape
    outputs = indices[0].numel() * 32 // width
    primary_indices = indices[0].view(outputs // 16, width // 64, 8, 4)
    grid = codebook.grid_packed_abs
    if rows == 1:
        output = torch.ops.quip_lib.decode_matvec_e8p(
            inputs[0], primary_indices, grid, outputs, width
        )
        if bits == 4:
            output = output + torch.ops.quip_lib.decode_matvec_e8p(
                inputs[0] / residual_scale,
                indices[1].view_as(primary_indices),
                grid,
                outputs,
                width,
            )
        return output.unsqueeze(0)
    if decode_batch is not None:
        output = decode_batch(inputs, primary_indices, grid)
        if bits == 4:
            output = output + decode_batch(
                inputs / residual_scale, indices[1].view_as(primary_indices), grid
            )
        return output.half()
    weights = quiptools_cuda.decompress_packed_e8p(primary_indices, grid)
    if bits == 4:
        weights = (
            weights
            + quiptools_cuda.decompress_packed_e8p(
                indices[1].view_as(primary_indices), grid
            )
            / residual_scale
        )
    return inputs @ weights.T
