"""Fused boundaries for factored randomized Hadamard transforms."""

from functools import lru_cache
from pathlib import Path

import torch

from .codebooks import decode_packed


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="harp_quip_rht",
        sources=[str(Path(__file__).parent / "csrc" / "rht.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        verbose=False,
    )


@lru_cache(maxsize=None)
def device_major(device):
    return torch.cuda.get_device_capability(device)[0]


def supported_factor(width, factor):
    if not isinstance(factor, int) or factor <= 0 or width % factor:
        return False
    subwidth = width // factor
    return 32 <= subwidth <= 8192 and subwidth & (subwidth - 1) == 0


def attach(handle, layer, stats, bits):
    """Attach quantized boundaries."""
    codebook = layer.codebook_class
    original = codebook.forward
    apply_boundary = None
    decode_batch = None
    empty_factor = torch.empty(0, device=layer.SU.device, dtype=torch.float32)

    def forward(module, inputs, indices, input_scale, output_scale, *args, **kwargs):
        nonlocal apply_boundary, decode_batch, empty_factor
        if module is not original.__self__:
            raise RuntimeError(
                "Remove acceleration before deepcopy; reinstall on the copy"
            )
        width, outputs = len(input_scale), len(output_scale)
        rows = inputs.numel() // width
        if (
            len(args) != 4
            or torch.is_grad_enabled()
            or layer.training
            or torch.is_autocast_enabled()
            or not inputs.is_cuda
            or inputs.dtype not in (torch.float16, torch.float32)
            or not 1 <= rows <= 8
            or module.scale != 32
            or kwargs.get("incoh_mode") != "had"
            or layer.incoh_mode != "had"
            or kwargs.get("train_mode", False)
            or kwargs.get("rescale_WH", False)
            or kwargs.get("rank", 0) > 0
            or input_scale.dtype != torch.float16
            or output_scale.dtype != torch.float16
        ):
            stats["calls"]["explicit_boundaries"] += 1
            return original(inputs, indices, input_scale, output_scale, *args, **kwargs)
        left, right, left_order, right_order = args
        if not supported_factor(width, left_order) or not supported_factor(
            outputs, right_order
        ):
            stats["calls"]["fallback_factor_geometry"] += 1
            return original(inputs, indices, input_scale, output_scale, *args, **kwargs)
        if apply_boundary is None:
            apply_boundary = extension().apply
        if empty_factor.device != inputs.device:
            empty_factor = torch.empty(0, device=inputs.device, dtype=torch.float32)
        left = left.float() if left is not None else empty_factor
        right = right.float() if right is not None else empty_factor
        transformed = apply_boundary(
            inputs.reshape(-1, width).contiguous(),
            input_scale.contiguous(),
            left,
            left_order,
            True,
            True,
        )
        use_shared = (
            handle.shared_core
            and rows > 1
            and not handle.native_core_only
            and device_major(inputs.device) >= 8
            and width < 65536
            and outputs < 65536
        )
        if use_shared and decode_batch is None:
            from .shared import decoder

            decode_batch = decoder()
        override = kwargs.get("resid_scale_override", -1)
        residual_scale = (
            override if override > 0 else getattr(module.codebook, "opt_resid_scale", 1)
        )
        product = decode_packed(
            transformed,
            indices,
            module.codebook,
            bits,
            residual_scale,
            decode_batch if use_shared else None,
        )
        result = apply_boundary(
            product.contiguous(),
            output_scale.contiguous(),
            right,
            right_order,
            False,
            inputs.dtype == torch.float16,
        )
        stats["calls"]["cuda_boundary"] += 1
        if use_shared:
            stats["calls"]["shared_packed_core"] += 1
        return result.reshape(*inputs.shape[:-1], outputs)

    handle._patch(codebook, "forward", forward)
    object.__setattr__(codebook, "_harp_runtime_owner", handle)
    from .integration import _MISSING

    handle._patches.append((codebook, "_harp_runtime_owner", _MISSING))
    stats["reason"] = "adapter_installed"
