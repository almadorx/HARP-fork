"""Measure fixed CUDA graphs and complete growing-cache generation requests."""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer
from eval.runtime import (
    add_arguments,
    generate,
    identity,
    load_model,
    memory,
    metadata,
    save,
    statistics,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_arguments(result)
    result.set_defaults(seed=246)
    result.add_argument("--compare_hf_path")
    result.add_argument(
        "--protocol", choices=("static", "generation"), default="static"
    )
    result.add_argument("--batch_size", type=int, default=1)
    result.add_argument("--seqlen", type=int, default=1)
    result.add_argument("--samples", type=int, default=512)
    result.add_argument("--rounds", type=int, default=21)
    result.add_argument("--max_rounds", type=int, default=41)
    result.add_argument("--warmup", type=int, default=32)
    result.add_argument("--requests", type=int, default=6)
    result.add_argument("--max_requests", type=int, default=24)
    result.add_argument("--batches", default="1,4")
    result.add_argument("--prompt_length", type=int, default=512)
    result.add_argument("--new_tokens", type=int, default=128)
    result.add_argument("--tokens_path")
    result.add_argument("--no_use_cuda_graph", action="store_true")
    result.add_argument("--output_path")
    return result


@torch.no_grad()
def main(args):
    if min(args.samples, args.rounds, args.requests, args.warmup) < 1:
        raise ValueError("Timing counts must be positive")
    paths = [args.hf_path] + ([args.compare_hf_path] if args.compare_hf_path else [])
    models = {}
    handles = {}
    report = dict(
        complete=False, metadata=metadata(args), setup=[], measurements=[], summaries=[]
    )
    graphs = args.protocol == "static" and not args.no_use_cuda_graph
    for index, path in enumerate(paths):
        local = argparse.Namespace(**vars(args))
        local.hf_path = path
        model, model_name, handle, setup = load_model(local, graphs=graphs)
        models[str(index)] = model
        handles[str(index)] = handle
        setup.update(label=str(index), path=path)
        report["setup"].append(setup)
        save(args.output_path, report)
    report["residency"] = (
        "one model"
        if len(models) == 1
        else "both models resident; sequential execution"
    )
    if args.protocol == "static":
        inputs = torch.ones(
            args.batch_size, args.seqlen, device="cuda", dtype=torch.long
        )
        for label, model in models.items():
            report["setup"][int(label)]["capture_memory_before"] = memory()
            start = time.perf_counter()
            model(inputs, use_cache=False)
            torch.cuda.synchronize()
            report["setup"][int(label)]["warm_capture_seconds"] = (
                time.perf_counter() - start
            )
            report["setup"][int(label)]["capture_memory_after"] = memory()
            if graphs and not model.built_graph:
                raise RuntimeError("CUDA graph capture did not occur")
        for repetition in range(max(args.rounds, args.max_rounds)):
            labels = list(models)
            random.shuffle(labels)
            for label in labels:
                model = models[label]
                token = model.static_args[0] if graphs else inputs
                for _ in range(args.warmup):
                    model(token, use_cache=False)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = memory()
                start_event, end_event = torch.cuda.Event(
                    enable_timing=True
                ), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                start_event.record()
                for _ in range(args.samples):
                    model(token, use_cache=False)
                end_event.record()
                end_event.synchronize()
                seconds = time.perf_counter() - start
                report["measurements"].append(
                    dict(
                        label=label,
                        round=repetition,
                        seconds=seconds,
                        tok_s=args.samples * inputs.numel() / seconds,
                        event_ms=start_event.elapsed_time(end_event),
                        memory_before=before,
                        memory_after=memory(),
                    )
                )
                save(args.output_path, report)
            if repetition + 1 >= args.rounds and (
                (repetition + 1 - args.rounds) % 10 == 0
                or repetition + 1 == args.max_rounds
            ):
                stats = {
                    label: statistics(
                        [
                            r["tok_s"]
                            for r in report["measurements"]
                            if r["label"] == label
                        ]
                    )
                    for label in models
                }
                report["summaries"].append(
                    dict(rounds=repetition + 1, statistics=stats)
                )
                save(args.output_path, report)
                if not any(value["needs_more"] for value in stats.values()):
                    break
    else:
        batches = [int(value) for value in args.batches.split(",")]
        if args.tokens_path:
            tokens = torch.load(args.tokens_path, map_location="cpu", weights_only=True)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
                raise ValueError(
                    "tokens_path must contain a two-dimensional token tensor"
                )
            report["data"] = identity(args.tokens_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            text = "It is a truth universally acknowledged that a person in possession of a good fortune must be in want of a companion. "
            row = tokenizer(text * 128, return_tensors="pt")["input_ids"]
            tokens = row.repeat(3 * max(batches), 1)
            report["data"] = dict(
                source="repeated text",
                text=text,
                warning="Pass --tokens_path to use a fixed, reproducible prompt tensor",
            )
        if tokens.shape[1] < args.prompt_length or tokens.shape[0] < 3 * max(batches):
            raise ValueError("Insufficient prompt rows or length")
        for batch in batches:
            prompts = [
                tokens[
                    i * max(batches) : i * max(batches) + batch, : args.prompt_length
                ]
                .cuda()
                .contiguous()
                for i in range(3)
            ]
            for label, model in models.items():
                torch.cuda.reset_peak_memory_stats()
                before = memory()
                start = time.perf_counter()
                generate(model, prompts[0], args.new_tokens)
                report.setdefault("warmup", []).append(
                    dict(
                        label=label,
                        batch=batch,
                        seconds=time.perf_counter() - start,
                        memory_before=before,
                        memory_after=memory(),
                    )
                )
            for repetition in range(max(args.requests, args.max_requests)):
                labels = list(models)
                random.shuffle(labels)
                for label in labels:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = memory()
                    record = generate(
                        models[label], prompts[repetition % 3], args.new_tokens
                    )
                    record.update(
                        label=label,
                        batch=batch,
                        round=repetition,
                        prompt_set=repetition % 3,
                        memory_before=before,
                        memory_after=memory(),
                    )
                    report["measurements"].append(record)
                    save(args.output_path, report)
                    print(
                        f"batch={batch} model={label} request={repetition+1} decode={record['decode_tok_s']:.2f} tok/s",
                        flush=True,
                    )
                if repetition + 1 >= args.requests and (
                    (repetition + 1 - args.requests) % 6 == 0
                    or repetition + 1 == args.max_requests
                ):
                    stats = {
                        label: {
                            key: statistics(
                                [
                                    r[key]
                                    for r in report["measurements"]
                                    if r["label"] == label and r["batch"] == batch
                                ]
                            )
                            for key in ("ttft_seconds", "decode_tok_s", "total_seconds")
                        }
                        for label in models
                    }
                    report["summaries"].append(
                        dict(batch=batch, requests=repetition + 1, statistics=stats)
                    )
                    save(args.output_path, report)
                    if not any(
                        value["needs_more"]
                        for result in stats.values()
                        for value in result.values()
                    ):
                        break
    report["coverage"] = {
        label: handle.report() if handle else None for label, handle in handles.items()
    }
    report["complete"] = True
    save(args.output_path, report)
    print(report["summaries"][-1], flush=True)


if __name__ == "__main__":
    main(parser().parse_args())
