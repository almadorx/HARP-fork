"QuIP# (+ HARP) quantization driver for Qwen3 models."

import argparse
import os
import time
from typing import Any, Dict

import glog

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune
from lib.linear import FusedLinear
from model.qwen3 import Qwen3DecoderLayer


from quantize_llama.quantize_finetune_llama import (
    make_harp_cfg_from_args,
    _save_layer_timing_summary,
    _log_timing_summary,
    _upload_layer_artifacts,
    _hf_complete_layers,
    wait_for_available_device,
    drain_active_workers,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--num_cpu_threads", default=8, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--devset_size", default=384, type=int)
parser.add_argument("--ctx_size", default=4096, type=int)
parser.add_argument("--save_path", type=str)
parser.add_argument("--hessian_path", type=str)
parser.add_argument("--base_model", default="Qwen/Qwen3-8B", type=str)
parser.add_argument("--sigma_reg", default=1e-2, type=float)
parser.add_argument("--sigma_reg2", default=1e-2, type=float)
parser.add_argument("--lora_rank", default=0, type=int, help="if <=0 then turned off")
parser.add_argument("--scale_override", default=-1, type=float)
parser.add_argument("--resid_scale_override", default=-1, type=float)
parser.add_argument("--codebook", type=str)
parser.add_argument("--quip_tune_iters", default=10, type=int)
parser.add_argument("--use_fp64", action="store_true")
parser.add_argument("--full_svd", action="store_true")
parser.add_argument("--no_use_buffered", action="store_true")
parser.add_argument("--rescale_WH", action="store_true")
parser.add_argument("--sample_proc", default=1, type=int)
parser.add_argument("--lowmem_ldlq", action="store_true")

parser.add_argument("--ft_lr", default=5e-5, type=float)
parser.add_argument("--ft_susv_lr", default=5e-4, type=float)
parser.add_argument("--ft_harp_lr", default=5e-7, type=float)
parser.add_argument("--ft_bs", default=8, type=int)
parser.add_argument("--ft_update_freq", default=2, type=int)
parser.add_argument("--ft_epochs", default=5, type=int)
parser.add_argument("--ft_valid_freq", default=1, type=int)
parser.add_argument("--ft_valid_size", default=128, type=float)
parser.add_argument("--ft_early_stop", default=3, type=int)
parser.add_argument("--ft_train_mode", action="store_true")
parser.add_argument("--ft_grad_ckpt", action="store_true")

parser.add_argument(
    "--xsamples_path",
    default=None,
    type=str,
    help="Folder containing x_samples files named like the hessian files (e.g. 0_qkv.pt)",
)

parser.add_argument(
    "--incoh_mode", default="had", type=str, choices=["had", "kron", "harp"]
)


parser.add_argument("--harp_b", default=8, type=int)
parser.add_argument("--harp_max_b", default=8, type=int)
parser.add_argument("--harp_passes", default=1, type=int)
parser.add_argument(
    "--harp_ordering_mode", default="stride", type=str, choices=["stride"]
)
parser.add_argument(
    "--harp_fixed_mixer",
    default="had_or_qr",
    type=str,
    choices=["had_or_ident", "identity", "had_or_qr"],
)
parser.add_argument("--harp_kron_fallback", action="store_true")
parser.add_argument("--harp_use_givens_b2", action="store_true")
parser.add_argument("--harp_steps", default=1200, type=int)
parser.add_argument("--harp_strategy", default="proxy", type=str, choices=["proxy"])
parser.add_argument("--harp_lr_u", default=3e-2, type=float)
parser.add_argument("--harp_lr_v", default=3e-2, type=float)
parser.add_argument("--harp_grad_clip", default=None, type=float)
parser.add_argument("--harp_reg_theta", default=0.0, type=float)
parser.add_argument("--harp_theta_init_scale", default=0.0, type=float)
parser.add_argument("--harp_theta_clip", default=None, type=float)
parser.add_argument("--harp_hbd_lambda", default=0.1, type=float)
parser.add_argument("--harp_hbd_block", default=8, type=int)
parser.add_argument("--harp_chunk_size", default=131072, type=int)
parser.add_argument(
    "--layer_idx",
    default=None,
    type=int,
    help="If set, only quantize this decoder layer index",
)
parser.add_argument(
    "--harp_hbd_mode",
    default="paper",
    type=str,
    choices=["paper", "ldl", "off"],
    help="H-structure term: paper=raw off-block penalty, "
    "ldl=scale-invariant tr(D)/tr(H) (LDLQ-aligned), off=none",
)
parser.add_argument(
    "--harp_eval_every",
    default=0,
    type=int,
    help="If >0, score the rotation every k steps (incl. the RHT-family init) "
    "with a scale-invariant surrogate and keep the best iterate.",
)
parser.add_argument(
    "--harp_q_recompute_every",
    default=1,
    type=int,
    help="Recompute Q(W_tilde) every k steps during HARP fitting. 1 = old behavior.",
)
parser.add_argument(
    "--harp_early_stop_window",
    default=0,
    type=int,
    help="Early-stop window for HARP. 0 disables early stopping.",
)
parser.add_argument(
    "--harp_early_stop_min_rel_improve",
    default=0.01,
    type=float,
    help="Minimum relative improvement required over the early-stop window.",
)

parser.add_argument("--resume_from_layer", default=0, type=int)
parser.add_argument("--stop_before_layer", default=None, type=int)
parser.add_argument("--skip_hf_existing", action="store_true")
parser.add_argument("--upload_after_layer", action="store_true")
parser.add_argument("--upload_repo_id", default=None, type=str)
parser.add_argument("--upload_repo_type", default="model", type=str)
parser.add_argument("--upload_token_env", default="HF_UPLOAD_TOKEN", type=str)
parser.add_argument(
    "--upload_lock_path", default="/tmp/harp_qwen3_hf_upload.lock", type=str
)
parser.add_argument("--upload_script", default=None, type=str)


def _make_causal_mask(
    config, batch_size, ctx_size, ref_hidden, past_key_values_length=0
):
    """4D causal mask; only applies the sliding window if it is actually
    enabled for this checkpoint (Qwen3-8B: use_sliding_window=False)."""
    if (
        getattr(config, "use_sliding_window", False)
        and getattr(config, "sliding_window", None) is not None
    ):
        return _prepare_4d_causal_attention_mask(
            None,
            (batch_size, ctx_size),
            ref_hidden,
            past_key_values_length,
            sliding_window=config.sliding_window,
        )
    return _prepare_4d_causal_attention_mask(
        None, (batch_size, ctx_size), ref_hidden, past_key_values_length
    )


def check_exist(idx, args):
    suffix = ["qkv", "o", "up", "down", "layernorm"]
    for _ in suffix:
        test = f"{args.save_path}/{idx}_{_}.pt"
        if not os.path.exists(test):
            return False
    return True


def quantize_qwen3_layer(
    layer, idx, cb, args, device, pre_orig_emb, orig_emb, model_config
):
    if check_exist(idx, args):
        glog.info(f"layer {idx} already complete locally")
        return

    layer_t0 = time.perf_counter()

    mixed_layer = Qwen3DecoderLayer(model_config, idx).cpu()
    with torch.no_grad():

        weights = [
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight,
        ]
        assert (
            layer.self_attn.q_proj.bias is None
        ), "Qwen3 attention projections are expected to be bias-free"

        fused_qkv_proj = FusedLinear(
            -1,
            [_.shape[0] for _ in weights],
            weights[0].shape[1],
            sum([_.shape[0] for _ in weights]),
            bias=False,
        )
        cur = 0
        for w in weights:
            fused_qkv_proj.weight[cur : cur + w.shape[0]].copy_(w)
            cur += w.shape[0]

        mixed_layer.self_attn.qkv_proj = fused_qkv_proj

        mixed_layer.self_attn.o_proj = layer.self_attn.o_proj

        mixed_layer.self_attn.q_norm.weight.copy_(layer.self_attn.q_norm.weight)
        mixed_layer.self_attn.k_norm.weight.copy_(layer.self_attn.k_norm.weight)

        weights = [layer.mlp.up_proj.weight, layer.mlp.gate_proj.weight]
        fused_upgate_proj = FusedLinear(
            -1,
            [_.shape[0] for _ in weights],
            weights[0].shape[1],
            sum([_.shape[0] for _ in weights]),
            bias=False,
        )
        cur = 0
        for w in weights:
            fused_upgate_proj.weight[cur : cur + w.shape[0]].copy_(w)
            cur += w.shape[0]
        mixed_layer.mlp.upgate_proj = fused_upgate_proj

        mixed_layer.mlp.down_proj = layer.mlp.down_proj

        mixed_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        mixed_layer.post_attention_layernorm.weight.copy_(
            layer.post_attention_layernorm.weight
        )

    timing_summary = finetune.quantize_finetune_decoder_layer(
        mixed_layer,
        [
            ("self_attn.qkv_proj", "qkv"),
            ("self_attn.o_proj", "o"),
            ("mlp.upgate_proj", "up"),
            ("mlp.down_proj", "down"),
        ],
        idx,
        cb,
        args,
        device,
        pre_orig_emb,
        orig_emb,
    )

    torch.save(
        {
            "input_layernorm": mixed_layer.input_layernorm.weight,
            "post_attention_layernorm": mixed_layer.post_attention_layernorm.weight,
            "q_norm": mixed_layer.self_attn.q_norm.weight,
            "k_norm": mixed_layer.self_attn.k_norm.weight,
        },
        f"{args.save_path}/{idx}_layernorm.pt",
    )
    del mixed_layer

    if timing_summary is not None:
        timing_summary["layer_wall_time_s"] = float(time.perf_counter() - layer_t0)
        glog.info(
            f"layer {idx} whole layer-block calibration time: "
            f'{timing_summary["layer_block_calibration_time_s"]:.2f}s'
        )
        glog.info(
            f"layer {idx} wall time (including wrapping/saving): "
            f'{timing_summary["layer_wall_time_s"]:.2f}s'
        )
        _save_layer_timing_summary(args.save_path, timing_summary)


@torch.no_grad()
def compute_layer_io(model, devset, layer_idx, batch_size, ctx_size, device):
    """Run the STOCK transformers Qwen3 layers up to `layer_idx` to compute
    the (pre, post) activations for that layer. Stock 4.51 Qwen3 layers
    require position_embeddings=(cos, sin)."""
    tok_emb = model.model.embed_tokens(devset)
    cur = tok_emb.to(device)

    rotary_emb = model.model.rotary_emb

    position_ids = torch.arange(ctx_size, dtype=torch.int64, device=device)[None, :]
    position_ids = position_ids.expand(batch_size, -1).contiguous()

    S = devset.shape[0]

    def _run_layer(layer, cur):
        out = []
        for j in range(0, S, batch_size):
            x = cur[j : j + batch_size]
            bsz = x.shape[0]
            pid = position_ids[:bsz]
            am = _make_causal_mask(model.config, bsz, ctx_size, x, 0)
            pe = rotary_emb(x, pid)
            out.append(
                layer(
                    x,
                    position_ids=pid,
                    attention_mask=am,
                    position_embeddings=pe,
                    use_cache=False,
                    output_attentions=False,
                )[0].detach()
            )
        return torch.cat(out, dim=0)

    for li in range(layer_idx):
        layer = model.model.layers[li].to(device)
        cur = _run_layer(layer, cur)
        layer.cpu()
        torch.cuda.empty_cache()

    pre = cur.cpu()

    layer = model.model.layers[layer_idx].to(device)
    post = _run_layer(layer, cur).cpu()
    layer.cpu()
    torch.cuda.empty_cache()

    return pre, post


def main(args):
    overall_t0 = time.perf_counter()
    spawn_ctx = mp.get_context("spawn")

    if args.incoh_mode == "harp":
        args.harp_cfg = make_harp_cfg_from_args(args)
    else:
        args.harp_cfg = None

    glog.info("==== Arguments ====")
    glog.info(f"{args}")

    cb = codebook.get_codebook(args.codebook)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype="auto", low_cpu_mem_usage=False
    )
    assert (
        model.config.model_type == "qwen3"
    ), f"expected a qwen3 model, got {model.config.model_type}"

    do_ft = int(args.ft_epochs) > 0

    all_config = {"quant_args": args, "model_config": model.config}
    quip_params = {
        "lora_rank": args.lora_rank,
        "rescale_WH": args.rescale_WH,
        "codebook": args.codebook,
        "codebook_version": cb.version,
        "codesz": cb.codesz,
        "idx_dtype": str(cb.idx_dtype),
        "packsz": cb.packsz,
        "incoh_mode": args.incoh_mode,
        "resid_scale_override": args.resid_scale_override,
        "quant_layers": [int(args.layer_idx)] if args.layer_idx is not None else None,
    }

    if args.incoh_mode == "harp":
        assert args.harp_cfg is not None
        quip_params["harp"] = dict(args.harp_cfg)

    all_config["model_config"].update({"quip_params": quip_params})
    torch.save(all_config, os.path.join(args.save_path, "config.pt"))

    glog.info("==== Full Config ====")
    glog.info(f"{all_config}")

    glog.info("loaded model")

    nproc = torch.cuda.device_count()

    devset = None
    orig_emb_cache = None
    position_ids = None
    attention_mask = None
    cur_device = 0
    proc_list = None
    active_procs = None

    if do_ft:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        devset = utils.sample_rp1t(
            tokenizer, args.devset_size, args.ctx_size, args.sample_proc
        )
        glog.info("loaded dataset and devset")

        orig_emb_cache = [model.model.embed_tokens(devset)]
        for _ in range(nproc):
            orig_emb_cache.append(
                torch.zeros(
                    orig_emb_cache[0].shape,
                    dtype=orig_emb_cache[0].dtype,
                    device=orig_emb_cache[0].device,
                )
            )

        position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[
            None, :
        ] + torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
        attention_mask = _make_causal_mask(
            model.config,
            args.batch_size,
            args.ctx_size,
            orig_emb_cache[0][: args.batch_size],
            0,
        )

        proc_list = [None for _ in range(nproc)]
    else:

        active_procs = {dev: None for dev in range(nproc)}
        active_layers = {}

    remote_done_layers = _hf_complete_layers(args, len(model.model.layers))

    start_layer = max(0, int(args.resume_from_layer))
    stop_before_layer = (
        len(model.model.layers)
        if args.stop_before_layer is None
        else min(int(args.stop_before_layer), len(model.model.layers))
    )

    layer_indices = [
        i for i in range(start_layer, stop_before_layer) if i not in remote_done_layers
    ]

    if do_ft and (args.resume_from_layer != 0 or args.skip_hf_existing):
        raise ValueError(
            "This resume/remote-skip path is intended for --ft_epochs 0. "
            "For ft_epochs > 0, use --layer_idx or implement activation replay carefully."
        )

    if args.layer_idx is not None:
        idx = int(args.layer_idx)
        assert 0 <= idx < len(model.model.layers)

        if idx in remote_done_layers:
            glog.info(f"layer {idx} already complete on HF; skipping")
            return

        if do_ft:
            pre_emb, post_emb = compute_layer_io(
                model, devset, idx, args.batch_size, args.ctx_size, device=0
            )
        else:
            pre_emb, post_emb = None, None

        quantize_qwen3_layer(
            model.model.layers[idx],
            idx,
            cb,
            args,
            device=0,
            pre_orig_emb=pre_emb,
            orig_emb=post_emb,
            model_config=all_config["model_config"],
        )

        total_wall_clock_s = time.perf_counter() - overall_t0
        _log_timing_summary(args.save_path, total_wall_clock_s)
        return

    if do_ft:

        rotary_emb = model.model.rotary_emb
        for i in range(len(model.model.layers)):
            glog.info(f"layer {i} gpu {cur_device}")
            if proc_list[cur_device] is not None:
                proc_list[cur_device].join()
                if cur_device == 0:
                    orig_emb_cache[0].copy_(orig_emb_cache[-1])
            if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
                proc_list[cur_device + 1].join()
            utils.clean()

            st = time.time()
            position_ids = position_ids.to(cur_device)
            attention_mask = attention_mask.to(cur_device)
            model.model.layers[i].to(cur_device)
            position_embeddings = None
            for j in range(args.devset_size // args.batch_size):
                x = orig_emb_cache[cur_device][
                    args.batch_size * j : args.batch_size * (j + 1)
                ].to(cur_device)
                if position_embeddings is None:

                    position_embeddings = rotary_emb(x, position_ids)
                orig_emb_cache[cur_device + 1][
                    args.batch_size * j : args.batch_size * (j + 1)
                ] = model.model.layers[i](
                    x,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False,
                )[
                    0
                ].cpu()
            model.model.layers[i].cpu()
            orig_msv = (
                orig_emb_cache[cur_device].float().norm() ** 2
                / orig_emb_cache[cur_device].numel()
            )
            target_msv = (
                orig_emb_cache[cur_device + 1].float().norm() ** 2
                / orig_emb_cache[cur_device + 1].numel()
            )
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            utils.clean()
            glog.info(
                "computed original embedding for layer {} in {}s, pre msv {}, post msv {}".format(
                    i, time.time() - st, orig_msv, target_msv
                )
            )

            proc_list[cur_device] = spawn_ctx.Process(
                target=quantize_qwen3_layer,
                args=(
                    model.model.layers[i],
                    i,
                    cb,
                    args,
                    cur_device,
                    orig_emb_cache[cur_device],
                    orig_emb_cache[cur_device + 1],
                    all_config["model_config"],
                ),
            )
            proc_list[cur_device].start()

            cur_device = (cur_device + 1) % nproc

        for p in proc_list:
            if p is not None:
                p.join()
    else:

        for i in layer_indices:
            device = wait_for_available_device(active_procs, active_layers, args)
            glog.info(f"layer {i} gpu {device}")
            utils.clean()

            p = spawn_ctx.Process(
                target=quantize_qwen3_layer,
                args=(
                    model.model.layers[i],
                    i,
                    cb,
                    args,
                    device,
                    None,
                    None,
                    all_config["model_config"],
                ),
            )
            p.start()
            active_procs[device] = p
            active_layers[device] = i
            glog.info(f"launched layer {i} on GPU {device}")

        drain_active_workers(active_procs, active_layers, args)

    total_wall_clock_s = time.perf_counter() - overall_t0
    _log_timing_summary(args.save_path, total_wall_clock_s)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    mp.set_sharing_strategy("file_system")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
