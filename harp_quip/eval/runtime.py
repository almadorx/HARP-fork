"""Shared loading, execution and measurement utilities."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lib.utils.unsafe_import import model_from_hf_path


class FixedGraphModel(torch.nn.Module):
    """Capture fixed-shape inference and recapture when its signature changes."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset()

    def reset(self):
        self.graph = None
        self.signature = None
        self.static_args = None
        self.static_kwargs = None
        self.static_output = None
        self.built_graph = False

    def forward(self, *args, **kwargs):
        if kwargs.get("use_cache", False):
            raise ValueError("Fixed graphs require use_cache=False")

        def signature(value):
            if isinstance(value, torch.Tensor):
                return (value.shape, value.dtype, value.device, value.stride())
            return value

        current = (
            tuple(map(signature, args)),
            tuple((k, signature(v)) for k, v in kwargs.items()),
        )
        if self.signature != current:
            self.reset()
            self.static_args = tuple(
                v.clone() if isinstance(v, torch.Tensor) else v for v in args
            )
            self.static_kwargs = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model(*self.static_args, **self.static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.static_output = self.model(*self.static_args, **self.static_kwargs)
            self.signature = current
            self.built_graph = True
        for target, source in zip(self.static_args, args):
            if isinstance(source, torch.Tensor):
                target.copy_(source)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                self.static_kwargs[key].copy_(value)
        self.graph.replay()
        return self.static_output


def add_arguments(parser):
    parser.add_argument("--hf_path", required=True)
    parser.add_argument(
        "--backend", choices=("optimized", "reference"), default="optimized"
    )
    parser.add_argument(
        "--shared_core", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prefill", choices=("auto", "reference", "fast"), default="auto"
    )
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_use_flash_attn", action="store_true", help=argparse.SUPPRESS
    )


def configure(args):
    device = torch.device(args.device_map)
    if device.type != "cuda":
        raise ValueError(
            "Evaluation requires a single CUDA device, e.g. --device_map cuda:0"
        )
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def load_model(args, graphs=False):
    configure(args)
    checkpoint = checkpoint_identity(args.hf_path)
    before = memory()
    start = time.perf_counter()
    model, model_name = model_from_hf_path(
        args.hf_path, use_cuda_graph=False, device_map=args.device_map
    )
    model.eval()
    handle = None
    if args.backend == "optimized":
        from lib.runtime import accelerate

        handle = accelerate(
            model,
            prefill={"auto": "auto", "reference": False, "fast": True}[args.prefill],
            shared_core=args.shared_core,
        )
    torch.cuda.synchronize()
    setup = dict(
        seconds=time.perf_counter() - start, memory_before=before, memory_after=memory()
    )
    setup["checkpoint"] = checkpoint
    if graphs:
        devices = {
            str(t.device)
            for t in list(model.parameters()) + list(model.buffers())
            if t.numel()
        }
        if devices != {f"cuda:{torch.cuda.current_device()}"}:
            raise ValueError(
                f"Fixed graphs require a model fully resident on the selected GPU; got {devices}"
            )
        model = FixedGraphModel(model)
    return model, model_name, handle, setup


def memory():
    free, total = torch.cuda.mem_get_info()
    return dict(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        device_used=total - free,
        device_total=total,
    )


def statistics(values, seed=246):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    if len(values) < 2:
        return dict(n=len(values), median=median, mean=median, needs_more=True)
    rng = np.random.default_rng(seed)
    bootstrap = np.median(
        rng.choice(values, size=(2000, len(values)), replace=True), axis=1
    )
    interval = np.quantile(bootstrap, [0.025, 0.975])
    split = len(values) // 2
    shift = float(abs(np.median(values[:split]) - np.median(values[split:])) / median)
    width = float((interval[1] - interval[0]) / median)
    return dict(
        n=len(values),
        median=median,
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        p10=float(np.quantile(values, 0.1)),
        p90=float(np.quantile(values, 0.9)),
        median_bootstrap95=interval.tolist(),
        relative_ci_width=width,
        half_median_shift=shift,
        needs_more=width > 0.05 or shift > 0.03,
    )


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return dict(
        path=str(path), bytes=Path(path).stat().st_size, sha256=digest.hexdigest()
    )


def checkpoint_identity(path):
    root = Path(path)
    if not root.is_dir():
        return {"model_id": path, "local_manifest": None}
    return {
        str(p.relative_to(root)): identity(p)
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix in {".json", ".bin", ".safetensors", ".model"}
    }


def metadata(args):
    versions = {}
    for name in ("torch", "transformers", "numpy", "triton", "lm-eval"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for directory in (
        root / "packages/harp_runtime/harp_runtime",
        root / "lib",
        root / "model",
        root / "eval",
    ):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cu", ".cuh"}:
                sources[str(path.relative_to(root))] = identity(path)["sha256"]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    result = dict(
        command=[sys.executable, *sys.argv],
        config=vars(args),
        python=platform.python_version(),
        versions=versions,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(torch.device(args.device_map)),
        capability=torch.cuda.get_device_capability(torch.device(args.device_map)),
        attention="sdpa",
        tf32=False,
        source_sha256=sources,
        git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
    )
    if args.backend == "optimized":
        from lib.runtime import accelerate
        import harp_runtime

        result["runtime"] = {
            "core_path": harp_runtime.__file__,
            "core_version": harp_runtime.__version__,
            "adapter_path": sys.modules[accelerate.__module__].__file__,
        }
    return result


def save(path, data):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")


@torch.no_grad()
def generate(model, inputs, outputs, select=None):
    """Measure prefill and growing-cache decode with no per-token host transfer."""
    if outputs < 2:
        raise ValueError("At least two output tokens are needed for decode timing")
    select = select or (lambda logits: logits[:, -1].argmax(-1, keepdim=True))
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = model(inputs, use_cache=True, num_logits_to_keep=1)
    token = select(result.logits)
    cache = result.past_key_values
    generated = [token]
    torch.cuda.synchronize()
    first = time.perf_counter()
    for _ in range(outputs - 1):
        result = model(
            token, past_key_values=cache, use_cache=True, num_logits_to_keep=1
        )
        token = select(result.logits)
        cache = result.past_key_values
        generated.append(token)
    torch.cuda.synchronize()
    end = time.perf_counter()
    cache_length = (
        cache.get_seq_length()
        if hasattr(cache, "get_seq_length")
        else cache[0][0].shape[-2]
    )
    if cache_length != inputs.shape[1] + outputs - 1:
        raise RuntimeError("Unexpected KV cache length")
    if not torch.isfinite(result.logits).all():
        raise RuntimeError("Nonfinite generation logits")
    return dict(
        ttft_seconds=first - start,
        decode_seconds=end - first,
        total_seconds=end - start,
        decode_tokens=(outputs - 1) * len(inputs),
        output_tokens=outputs * len(inputs),
        decode_tok_s=(outputs - 1) * len(inputs) / (end - first),
        output_tok_s=outputs * len(inputs) / (end - start),
        final_cache_length=int(cache_length),
        generated_ids=torch.cat(generated, dim=1).cpu().tolist(),
    )
