"""Load repository models without introducing package import cycles."""

import transformers

from model.llama import LlamaForCausalLM
from . import graph_wrapper


def _resolve_model_str(config, path):
    """Resolve the tokenizer source from the saved base-model metadata."""
    name = getattr(config, "_name_or_path", "")
    if name:
        return name
    parameters = getattr(config, "quip_params", {})
    return parameters.get("base_model") or path


def model_from_hf_path(
    path, use_cuda_graph=True, use_flash_attn=True, device_map="auto"
):
    """Load packed Llama/Qwen3 checkpoints or ordinary Hugging Face models."""
    config = transformers.AutoConfig.from_pretrained(path)
    if hasattr(config, "quip_params"):
        if config.model_type == "llama":
            config_class = transformers.LlamaConfig
            model_class = LlamaForCausalLM
        elif config.model_type == "qwen3":
            from model.qwen3 import Qwen3ForCausalLM

            config_class = transformers.Qwen3Config
            model_class = Qwen3ForCausalLM
        else:
            raise ValueError(f"Unsupported quantized model type: {config.model_type}")
        model_config = config_class.from_pretrained(path)
        model_name = _resolve_model_str(model_config, path)
    else:
        model_name = path
        model_class = transformers.AutoModelForCausalLM

    if use_cuda_graph:
        model_class = graph_wrapper.get_graph_wrapper(model_class)
    model = model_class.from_pretrained(
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    return model, model_name
