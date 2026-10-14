"""Evaluate zero-shot tasks with optimized HARP or RHT execution."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lm_eval import evaluator
from transformers import AutoTokenizer
from lib.utils import LMEvalAdaptor
from eval.runtime import add_arguments, load_model, memory, metadata, save


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_arguments(result)
    result.add_argument("--batch_size", type=int, default=1)
    result.add_argument("--tasks", required=True)
    result.add_argument("--num_fewshot", type=int, default=0)
    result.add_argument("--max_length", type=int, default=-1)
    result.add_argument("--limit", type=int)
    result.add_argument(
        "--no_use_cuda_graph", action="store_true", help=argparse.SUPPRESS
    )
    result.add_argument("--output_path")
    return result


@torch.no_grad()
def main(args):
    model, model_name, handle, setup = load_model(args, graphs=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    adapter = LMEvalAdaptor(
        model_name, model, tokenizer, args.batch_size, max_length=args.max_length
    )
    warm_memory = memory()
    warm_start = time.perf_counter()
    model(torch.ones(1, 1, device="cuda", dtype=torch.long), use_cache=False)
    torch.cuda.synchronize()
    warmup = dict(
        seconds=time.perf_counter() - warm_start,
        memory_before=warm_memory,
        memory_after=memory(),
    )
    torch.cuda.reset_peak_memory_stats()
    before = memory()
    results = evaluator.simple_evaluate(
        model=adapter,
        tasks=args.tasks.split(","),
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    print(evaluator.make_table(results))
    results["config"]["model"] = args.hf_path
    results["runtime"] = dict(
        metadata=metadata(args),
        setup=setup,
        warmup=warmup,
        memory_before=before,
        memory_after=memory(),
        coverage=handle.report() if handle else None,
        graphs=False,
        limited=args.limit is not None,
    )
    save(args.output_path, results)


if __name__ == "__main__":
    main(parser().parse_args())
