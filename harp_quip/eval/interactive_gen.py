"""Generate text using optimized HARP or RHT inference."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer
from eval.runtime import add_arguments, load_model, memory


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_arguments(result)
    result.add_argument("--prompt")
    result.add_argument("--streaming", action="store_true")
    result.add_argument("--max_new_tokens", type=int, default=128)
    result.add_argument("--top_k", type=int, default=32)
    result.add_argument("--temperature", type=float, default=0.6)
    result.add_argument("--num_samples", type=int, default=1)
    result.add_argument("--compile", action="store_true", help=argparse.SUPPRESS)
    return result


@torch.no_grad()
def main(args):
    if args.compile:
        raise ValueError(
            "torch.compile is not qualified for this runtime; omit --compile"
        )
    model, model_name, handle, _ = load_model(args, graphs=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    while True:
        prompt = (
            args.prompt if args.prompt is not None else input("Prompt (quit to exit): ")
        )
        if prompt == "quit":
            break
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [dict(role="user", content=prompt)],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        options = dict(
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_samples,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        if args.temperature > 0:
            options.update(temperature=args.temperature, top_k=args.top_k)
        if args.streaming:
            from transformers import TextStreamer

            if args.num_samples != 1:
                raise ValueError("Streaming requires num_samples=1")
            options["streamer"] = TextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        torch.cuda.reset_peak_memory_stats()
        outputs = model.generate(**inputs, **options)
        if not args.streaming:
            for output in outputs:
                print(
                    tokenizer.decode(
                        output[inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                )
        print("Peak allocated GiB:", memory()["peak_allocated"] / 2**30)
        if args.prompt is not None:
            break
    if handle:
        print(handle.report())


if __name__ == "__main__":
    main(parser().parse_args())
