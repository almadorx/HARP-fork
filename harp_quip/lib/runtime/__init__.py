"""Optimized inference for HARP and RHT QuIP projections."""

import sys
from pathlib import Path

source = Path(__file__).resolve().parents[2] / "packages" / "harp_runtime"
if source.is_dir():
    sys.path.insert(0, str(source))

import harp_runtime

if (
    source.is_dir()
    and Path(harp_runtime.__file__).resolve().parent != source / "harp_runtime"
):
    raise ImportError("Restart Python to load the HARP core from this checkout")

if getattr(harp_runtime, "__version__", None) != "0.2.0":
    raise ImportError(
        "This adapter requires harp-runtime 0.2.0; restart after loading the matching core"
    )

from .integration import accelerate

__all__ = ["accelerate"]
