"""Packed E8P matrix multiplication for small batches."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def decoder():
    from torch.utils.cpp_extension import load

    return load(
        name="harp_quip_e8p",
        sources=[str(Path(__file__).parent / "csrc" / "e8p.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    ).decode_batch_e8p
