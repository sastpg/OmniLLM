from abc import abstractmethod

import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Type, Union, List
from api.api import Message



class AbstractModel:
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map = device_map,
        )
    
    @abstractmethod
    def __call__(self, messages: List[Message], temperature: float=0.6, top_p: float=0.9, do_sample: bool=True, max_new_tokens: int=512) -> str:
        raise NotImplementedError


