from abc import ABC, abstractmethod

import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers.generation.stopping_criteria import StoppingCriteria, STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
from typing import Union, List, Dict

_TOOL_PROMPT = """You have access to the following set of tools. Here are the names and descriptions for each tool:

{tools}

Given the following question, return the name and input of the tool to use(a JSON blob with 'name' and 'arguments' keys). If it doesn't require any of the provided tools to answer, just return an empty JSON blob {{}}.

Note The `arguments` should be a dictionary, with keys corresponding to the argument names and the values corresponding to the requested values.

Qustion: {query}"""

class StopAtTokens(StoppingCriteria):
    def __init__(self, token_id_list: list[int] = None):
        self.token_id_list = token_id_list
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


class AbstractModel(ABC):
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map = device_map,
            trust_remote_code=True
        )
    
    @abstractmethod
    def __call__(self, messages: List[Dict[str, str]], tools: list=None, stream=False, **kwargs) -> str:
        raise NotImplementedError


def process_input(history: list[dict], tools: list) -> list[dict]:
    chat_messages = history[:-1]
    chat_messages.append(dict(role="user", content=_TOOL_PROMPT.format(tools="\n".join(tools), query=history[-1]["content"])))
    return chat_messages