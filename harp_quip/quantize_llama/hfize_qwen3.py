"Export a QuIP#/HARP-quantized Qwen3 checkpoint (produced by"

import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.qwen3 import Qwen3ForCausalLM
from lib.utils.model_version import MODEL_VERSION

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--quantized_path", type=str)
parser.add_argument("--hf_output_path", type=str)
parser.add_argument("--smoke_tokens", type=int, default=8)
parser.add_argument(
    "--quantize_harp_theta",
    action="store_true",
    help="Quantize HARP theta parameters to int8 (packed upper triangle) before bit counting/saving.",
)


def bits_per_weight_from_codebook(codebook_name: str) -> int:
    mapping = {
        "E8P12": 2,
        "E8P12RVQ3B": 3,
        "E8P12RVQ4B": 4,
    }
    if codebook_name not in mapping:
        raise ValueError(f"Unknown codebook for BPP accounting: {codebook_name}")
    return mapping[codebook_name]


def count_bits_for_quant_linear(
    mod, bits_per_weight: int, quantize_harp_theta: bool = False
):
    m = mod.out_features
    n = mod.in_features
    num_params = m * n
    bits_quant = num_params * bits_per_weight

    harp = getattr(mod, "harp", None)
    if harp is not None and quantize_harp_theta:
        qfn = getattr(harp, "quantize_theta_8bit", None)
        if callable(qfn):
            qfn()
    bits_harp = harp.get_storage_bits() if harp is not None else 0
    return num_params, bits_quant + bits_harp


def main(args):
    total_params = 0
    total_bits = 0

    if os.path.exists(args.hf_output_path):
        raise FileExistsError(f"Use a new export directory: {args.hf_output_path}")
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(
        os.path.join(args.quantized_path, "config.pt"), weights_only=False
    )
    model_config = saved_config["model_config"]
    print("Model Config:", model_config)

    assert (
        model_config.model_type == "qwen3"
    ), f"hfize_qwen3 expects a qwen3 config, got {model_config.model_type}"

    codebook_id = codebook.get_id(model_config.quip_params["codebook"])
    codesz = model_config.quip_params["codesz"]

    bits_per_weight = bits_per_weight_from_codebook(
        model_config.quip_params["codebook"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model_config.quip_params["model_version"] = MODEL_VERSION

    model_config.quip_params["base_model"] = model_config._name_or_path
    model = Qwen3ForCausalLM.from_pretrained(
        model_config._name_or_path,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        config=model_config,
    ).half()

    cpu = torch.device("cpu")

    if os.path.exists(f"{args.quantized_path}/lmhead.pt"):
        lmhead_data = torch.load(
            f"{args.quantized_path}/lmhead.pt", map_location=cpu, weights_only=False
        )
        model.lm_head.weight.copy_(lmhead_data["lm_head"])
        model.model.norm.weight.copy_(lmhead_data["norm"])

    qp = model_config.quip_params
    quant_layers = qp.get("quant_layers", None)
    quant_layers = set(quant_layers) if quant_layers is not None else None

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]

        if os.path.exists(f"{args.quantized_path}/{ii}_layernorm.pt"):
            ln_data = torch.load(
                f"{args.quantized_path}/{ii}_layernorm.pt",
                map_location=cpu,
                weights_only=False,
            )
            layer.input_layernorm.weight.copy_(ln_data["input_layernorm"])
            layer.post_attention_layernorm.weight.copy_(
                ln_data["post_attention_layernorm"]
            )

            if "q_norm" in ln_data:
                layer.self_attn.q_norm.weight.copy_(ln_data["q_norm"])
            if "k_norm" in ln_data:
                layer.self_attn.k_norm.weight.copy_(ln_data["k_norm"])

        if quant_layers is not None and (ii not in quant_layers):
            glog.info(f"skipping quant load for layer {ii} (dense)")
            continue

        saved_layer = torch.load(
            f"{args.quantized_path}/{ii}_qkv.pt", map_location=cpu, weights_only=False
        )
        for i in range(len(saved_layer["scales"])):
            layer.self_attn.qkv_proj.fuse_scales[i].copy_(saved_layer["scales"][i])
        utils.unpack_quip(layer.self_attn.qkv_proj, saved_layer, codebook_id, codesz)

        nparams, nbits = count_bits_for_quant_linear(
            layer.self_attn.qkv_proj,
            bits_per_weight,
            quantize_harp_theta=args.quantize_harp_theta,
        )
        total_params += nparams
        total_bits += nbits

        saved_layer = torch.load(
            f"{args.quantized_path}/{ii}_o.pt", map_location=cpu, weights_only=False
        )
        utils.unpack_quip(layer.self_attn.o_proj, saved_layer, codebook_id, codesz)

        nparams, nbits = count_bits_for_quant_linear(
            layer.self_attn.o_proj,
            bits_per_weight,
            quantize_harp_theta=args.quantize_harp_theta,
        )
        total_params += nparams
        total_bits += nbits

        saved_layer = torch.load(
            f"{args.quantized_path}/{ii}_up.pt", map_location=cpu, weights_only=False
        )
        for i in range(len(saved_layer["scales"])):
            layer.mlp.upgate_proj.fuse_scales[i].copy_(saved_layer["scales"][i])
        utils.unpack_quip(layer.mlp.upgate_proj, saved_layer, codebook_id, codesz)
        nparams, nbits = count_bits_for_quant_linear(
            layer.mlp.upgate_proj,
            bits_per_weight,
            quantize_harp_theta=args.quantize_harp_theta,
        )
        total_params += nparams
        total_bits += nbits

        saved_layer = torch.load(
            f"{args.quantized_path}/{ii}_down.pt", map_location=cpu, weights_only=False
        )
        utils.unpack_quip(layer.mlp.down_proj, saved_layer, codebook_id, codesz)
        nparams, nbits = count_bits_for_quant_linear(
            layer.mlp.down_proj,
            bits_per_weight,
            quantize_harp_theta=args.quantize_harp_theta,
        )
        total_params += nparams
        total_bits += nbits
        glog.info(f"loaded layer {ii} down")

    glog.info(f"saving model...")
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    tokenizer.save_pretrained(args.hf_output_path)

    del model

    model, _ = model_from_hf_path(
        args.hf_output_path, use_cuda_graph=False, device_map="cuda:0"
    )
    from lib.runtime import accelerate

    model.eval()
    runtime = accelerate(model, shared_core=True)

    glog.info("successfully loaded hfized model")
    glog.info("generating some text...")

    start = time.time()
    prompt = "It is a truth universally acknowledged that"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        max_new_tokens=args.smoke_tokens,
        return_dict_in_generate=True,
    )
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    glog.info(output_str)

    bpp = total_bits / total_params
    glog.info(
        "==== Projection-only storage estimate (not complete deployment BPP) ===="
    )
    glog.info(f"Total quantized parameters: {total_params:,}")
    glog.info(f"Total bits (quantized + HARP): {total_bits:,}")
    glog.info(f"Partial estimated bits per quantized projection weight: {bpp:.4f}")
    glog.info(f"elapsed: {time.time() - start}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
