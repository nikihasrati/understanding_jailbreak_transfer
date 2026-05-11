
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from pipeline.model_utils.model_base import ModelBase

PHI3_CHAT_TEMPLATE = """<|user|>
{instruction}<|end|>
<|assistant|>
"""

def format_instruction_phi3(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    formatted_instruction = PHI3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_phi(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_phi3(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_phi3(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

class Phi3Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self, system=None):
        return functools.partial(tokenize_instructions_phi, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)
    
    def _get_model_block_modules(self):
        return self.model.model.layers
