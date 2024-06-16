import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List
from api.api import Message

class Llama3:
    def __init__(self, model_path: Path, device_map: Union[dict,str]=''):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map = device_map,
        )
    
    def __call__(self, messages: List[Message], temperature: float=0.6, top_p: float=0.9, do_sample: bool=True, max_new_tokens: int=512) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            #repetition_penalty=1.3, 
        )
        if len(outputs) > 0:
            response = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True)
        else:
            return None

