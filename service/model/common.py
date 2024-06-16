from abc import abstractmethod

import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List, Dict


class AbstractModel:
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map = device_map,
            trust_remote_code=True
        )
    
    @abstractmethod
    def __call__(self, messages: List[Dict[str, str]], temperature: float, top_p: float, do_sample: bool, max_new_tokens: int) -> str:
        raise NotImplementedError


