"""Staged transforms with fused CUDA execution and a differentiable reference."""

from functools import lru_cache
from pathlib import Path
import math
import importlib.util
import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="harp_runtime_rotation",
        sources=[str(Path(__file__).parent / "csrc" / "rotation.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        verbose=False,
    )


def reference_stage(x, k, stride, inverse=False):
    """Differentiable gather/batched-matmul/scatter reference for any radix."""
    rows, n = x.shape
    b = k.shape[-1]
    c = n // (b * stride)
    v = x.reshape(rows, c, b, stride).transpose(2, 3)
    w = k.reshape(c, stride, b, b)
    y = torch.matmul(v.unsqueeze(-2), w if inverse else w.transpose(-1, -2))
    return y.squeeze(-2).transpose(2, 3).reshape(rows, n)


def specialized_reduction(b, stride):
    return (
        (b == 8 and stride in (1, 2, 4, 8, 16, 64, 512))
        or (b == 43 and stride in (256, 512))
        or (b in (3, 4) and stride >= 32)
    )


class RotationPlan:
    """Execute complete mixed-radix passes of coefficient blocks."""

    def __init__(self, passes, *, prefill=True, max_fiber=512):
        if not passes or any(not p for p in passes):
            raise ValueError("At least one nonempty complete pass is required")
        self.passes = tuple(tuple(p) for p in passes)
        first = self.passes[0][0]
        if first.ndim != 3 or first.shape[1] != first.shape[2]:
            raise ValueError("Blocks must have shape [n/radix,radix,radix]")
        self.n = first.shape[0] * first.shape[1]
        self.device = first.device
        if self.n < 1 or self.n >= 2**27:
            raise ValueError("Unsupported dimension; requires 0 < n < 2**27")
        if not 1 <= max_fiber <= 1024:
            raise ValueError("max_fiber must be between1 and1024")
        self.prefill = prefill
        self.cuda_major = (
            torch.cuda.get_device_capability(first.device)[0] if first.is_cuda else 0
        )
        self.tensorcore_prefill = (
            self.cuda_major >= 8 and importlib.util.find_spec("triton") is not None
        )
        self.stages, self.paired = [], []
        for p in self.passes:
            s = 1
            for t, k in enumerate(p):
                b = k.shape[-1]
                if (
                    k.ndim != 3
                    or b < 1
                    or k.shape != (self.n // b, b, b)
                    or self.n % (s * b)
                    or k.device != self.device
                    or k.dtype not in (torch.float32, torch.float64)
                ):
                    raise ValueError(
                        "Inconsistent block shape, device, dtype or schedule"
                    )
                self.stages.append((k, s))
                pair = (
                    b > 1
                    and t + 1 < len(p)
                    and p[t + 1].shape[-1] > 1
                    and b * p[t + 1].shape[-1] <= max_fiber
                    and (
                        b * p[t + 1].shape[-1] <= 256
                        or (
                            specialized_reduction(b, s)
                            and specialized_reduction(p[t + 1].shape[-1], s * b)
                        )
                    )
                    and not (t and self.paired[-1])
                )
                self.paired.append(pair)
                s *= b
            if s != self.n:
                raise ValueError("The radix product in each pass must equal n")
        self.fused_eligible = first.is_cuda and all(
            k.dtype == torch.float32
            and k.is_contiguous()
            and k.numel() < 2**31
            and k.shape[-1] <= 256
            for k, _ in self.stages
        )
        self._cpp = None
        self._empty_scale = torch.empty(0, device=self.device, dtype=torch.float16)
        self.last_path = "unexecuted"

    def prepare(self):
        """Build the CUDA plan before graph capture; no full-model warmup needed."""
        if self.fused_eligible and self._cpp is None:
            self._cpp = extension().RotationPlan(
                [k.contiguous() for k, _ in self.stages],
                [s for _, s in self.stages],
                self.paired,
            )
        return self

    def apply(self, x, inverse=False):
        if x.shape[-1] != self.n or x.device != self.device:
            raise ValueError("Input dimension/device must match the plan")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("Standalone plans require FP32 or FP64 input")
        shape = x.shape
        y = x.reshape(-1, self.n).contiguous()
        differentiable = torch.is_grad_enabled()
        if (
            not differentiable
            and y.dtype == torch.float32
            and self.fused_eligible
            and 0 < len(y) <= 8
        ):
            self.prepare()
            self.last_path = "cuda_fused"
            return self._cpp.apply(y, self._empty_scale, inverse, 0, 1.0).reshape(shape)
        if (
            not differentiable
            and y.is_cuda
            and y.dtype == torch.float32
            and self.prefill
            and self.tensorcore_prefill
            and len(y) > 8
            and all(
                k.dtype == torch.float32 and k.shape[-1] <= 128 for k, _ in self.stages
            )
        ):
            from .prefill import apply

            self.last_path = "autosort_prefill" if len(y) >= 128 else "grouped_prefill"
            return apply(y, self.stages, inverse).reshape(shape)
        self.last_path = "autograd_reference" if differentiable else "reference"
        for k, s in reversed(self.stages) if inverse else self.stages:
            y = reference_stage(y, k.to(dtype=y.dtype), s, inverse)
        return y.reshape(shape)

    def boundary(self, x, scale, *, inverse=False, mode=1, multiplier=1.0):
        """Apply input scaling or output scaling with a fused transform."""
        if x.shape[-1] != self.n or x.device != self.device:
            raise ValueError("Input dimension/device must match the plan")
        if not self.fused_eligible or not 0 < x.numel() // self.n <= 8:
            raise ValueError(
                "Fused boundary requires CUDA, small rows, contiguous radices<=256"
            )
        if x.dtype not in (torch.float16, torch.float32) or mode not in (1, 2, 3):
            raise ValueError("Unsupported boundary input dtype/mode")
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Positive finite multiplier required")
        if torch.is_grad_enabled() or torch.is_autocast_enabled():
            raise RuntimeError("Boundary fusion is inference-only")
        self.prepare()
        self.last_path = "cuda_boundary"
        return self._cpp.apply(
            x.reshape(-1, self.n).contiguous(),
            scale.contiguous(),
            inverse,
            mode,
            multiplier,
        ).reshape(x.shape)

    def description(self):
        return dict(
            dimension=self.n,
            device=str(self.device),
            passes=[[k.shape[-1] for k in p] for p in self.passes],
            fused_eligible=self.fused_eligible,
            specialized_reduction=all(
                specialized_reduction(k.shape[-1], s) for k, s in self.stages
            ),
            operations=len(self.stages) - sum(self.paired),
            coefficient_bytes=sum(k.numel() * k.element_size() for k, _ in self.stages),
            last_path=self.last_path,
        )
