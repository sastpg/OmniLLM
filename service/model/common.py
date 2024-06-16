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
    def __call__(self, messages: List[Dict[str, str]], temperature: float=0.6, top_p: float=0.9, do_sample: bool=True, max_new_tokens: int=512) -> str:
        raise NotImplementedError


