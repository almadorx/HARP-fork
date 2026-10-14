"""Evaluate token-weighted perplexity with optimized quantized inference."""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from lib.utils import gptq_data_utils
from eval.runtime import add_arguments, identity, load_model, memory, metadata, save


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_arguments(result)
    result.add_argument("--seqlen", type=int, default=4096)
    result.add_argument("--datasets", default="wikitext2,c4")
    result.add_argument("--tokens_path")
    result.add_argument("--limit", type=int)
    result.add_argument("--no_use_cuda_graph", action="store_true")
    result.add_argument("--output_path")
    return result


@torch.no_grad()
def main(args):
    if args.seqlen < 2:
        raise ValueError("seqlen must be at least two")
    model, model_name, handle, setup = load_model(
        args, graphs=not args.no_use_cuda_graph
    )
    report = dict(complete=False, metadata=metadata(args), setup=setup, datasets=[])
    names = ["tokens_file"] if args.tokens_path else args.datasets.split(",")
    for name in names:
        if args.tokens_path:
            tokens = torch.load(args.tokens_path, map_location="cpu", weights_only=True)
            if isinstance(tokens, dict):
                tokens = tokens["input_ids"]
            source = identity(args.tokens_path)
        else:
            tokens = gptq_data_utils.get_test_tokens(
                name, seed=args.seed, seqlen=args.seqlen, model=model_name
            )
            source = dict(dataset=name, seed=args.seed, tokenizer=model_name)
        count = tokens.numel() // args.seqlen
        if args.limit is not None:
            count = min(count, args.limit)
        if count < 1:
            raise ValueError("Not enough tokens for one evaluation chunk")
        chunks = tokens.reshape(-1)[: count * args.seqlen].reshape(count, args.seqlen)
        first = chunks[:1].cuda()
        warm_memory = memory()
        warm_start = time.perf_counter()
        model(
            first, use_cache=False, output_hidden_states=False, output_attentions=False
        )
        torch.cuda.synchronize()
        warmup = dict(
            seconds=time.perf_counter() - warm_start,
            memory_before=warm_memory,
            memory_after=memory(),
        )
        torch.cuda.reset_peak_memory_stats()
        before = memory()
        losses = []
        for ids in tqdm(chunks, desc=name):
            ids = ids[None].cuda()
            logits = model(
                ids,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            ).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1),
            )
            losses.append(float(loss))
        values = torch.tensor(losses, dtype=torch.float64)
        mean = float(values.mean())
        standard_error = (
            float(values.std(correction=1) / math.sqrt(count)) if count > 1 else None
        )
        result = dict(
            dataset=name,
            source=source,
            chunks=count,
            predicted_tokens=count * (args.seqlen - 1),
            seqlen=args.seqlen,
            mean_nll=mean,
            perplexity=math.exp(mean),
            chunk_nll=losses,
            nll_standard_error=standard_error,
            ppl_ci95=(
                [
                    math.exp(mean - 1.96 * standard_error),
                    math.exp(mean + 1.96 * standard_error),
                ]
                if standard_error is not None
                else None
            ),
            memory_before=before,
            memory_after=memory(),
            warmup=warmup,
            limited=args.limit is not None,
        )
        report["datasets"].append(result)
        save(args.output_path, report)
        print(
            f"{name}: PPL={result['perplexity']:.6f}, NLL={mean:.6f}, chunks={count}",
            flush=True,
        )
    report["coverage"] = handle.report() if handle else None
    report["complete"] = True
    save(args.output_path, report)


if __name__ == "__main__":
    main(parser().parse_args())
