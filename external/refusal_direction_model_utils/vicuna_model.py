import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase


# Chat templates from: https://huggingface.co/junelee/wizard-vicuna-13b/discussions/1
VICUNA_DEFAULT_SYSTEM_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions."""

VICUNA_CHAT_TEMPLATE = "### Human:\n{instruction}\n### Assistant:\n"

VICUNA_CHAT_TEMPLATE_WITH_SYSTEM = "{system_prompt}\n\n### Human:\n{instruction}\n### Assistant:\n"

VICUNA_REFUSAL_TOKS = [29902, 2887] # ['I', 'As']

def format_instruction_vicuna_chat(
    instruction: str,
    system: str=None,
    output: str=None,
    include_trailing_whitespace=True
):
    if system is not None:
        if system == "default":
            system = VICUNA_DEFAULT_SYSTEM_PROMPT
        formatted_instruction = VICUNA_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
    else:
        formatted_instruction = VICUNA_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_vicuna_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_vicuna_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_vicuna_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_vicuna_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_vicuna_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)

class VicunaModel(ModelBase):
    def _load_model(self, model_path, dtype=torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_vicuna_chat, tokenizer=self.tokenizer)
    
    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_eoi_toks(self):
        return self.tokenizer.encode(VICUNA_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)
    
    def _get_refusal_toks(self):
        return VICUNA_REFUSAL_TOKS

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])
    
    def _get_orthogonalization_mod_fn(self, direction):
        return functools.partial(orthogonalize_vicuna_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction, coeff, layer):
        return functools.partial(act_add_vicuna_weights, direction=direction, coeff=coeff, layer=layer)
