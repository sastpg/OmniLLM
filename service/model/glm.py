import torch
from pathlib import Path

from typing import Union, List, Dict
from model.common import AbstractModel, process_input


class GLM(AbstractModel):
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        super().__init__(model_path, device_map)

    def __call__(self, messages: List[Dict[str, str]], tools: list=None, stream=False, **kwargs) -> str:
        if tools:
            messages = process_input(messages, tools)
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = inputs.to(self.model.device)
        # model = self.model.to(self.model.device).eval()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **kwargs
            )
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)