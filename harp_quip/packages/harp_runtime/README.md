# HARP transform runtime

Execute staged HARP transforms on CPUs and NVIDIA GPUs. The package contains transform planning, CUDA fusion, Triton prefill and a differentiable reference.

## Installation

```sh
python -m pip install --no-build-isolation './packages/harp_runtime[cuda]'
```

CUDA execution needs CUDA-enabled PyTorch, the CUDA toolkit including `nvcc`, a C++ compiler and Ninja. Kernels compile on first use and are cached by PyTorch. Fast prefill additionally needs Triton and an Ampere or newer GPU. CPU and FP64 execution use PyTorch. Tested with Python 3.10, PyTorch 2.8, CUDA 12.9 and Triton 3.4 on an RTX 5080.

## Transform API

```python
import torch
from harp_runtime import RotationPlan

plan = RotationPlan([[stage0, stage1, stage2]])
with torch.no_grad():
    output = plan.apply(inputs)
    adjoint = plan.apply(inputs, inverse=True)
```

Each stage is a tensor of shape `[dimension / radix, radix, radix]`. The product of the radices in every pass must equal `dimension`. Inputs and coefficients use FP32 or FP64. The transpose reverses the stage order.

`plan.description()` reports dimensions and the last executed path. Call `plan.prepare()` before CUDA graph capture. Recapture CUDA graphs after changes. Noncontiguous blocks use an uncached path.

`plan.boundary()` combines a transform with per-channel scaling and FP16/FP32 conversion for small rows. Mode 1 scales inputs and divides the output by the positive scalar `multiplier`; modes 2 and 3 apply the transform then multiply by channel scales and `multiplier`, returning FP16 or FP32 respectively. The default multiplier is one. This operation is inference-only.

## Integrating a quantizer

Supply expanded transform blocks to `RotationPlan` and use its outputs around your quantized matrix multiplication. This repository's QuIP#-specific adapter is `lib.runtime.accelerate`.
