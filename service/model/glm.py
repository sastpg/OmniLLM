import torch
from pathlib import Path

from typing import Union, List, Dict
from model.common import AbstractModel


class GLM(AbstractModel):
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        super().__init__(model_path, device_map)

    def __call__(self, messages: List[Dict[str, str]], temperature: float = 0.7, top_p: float = 0.7, do_sample: bool = True, max_new_tokens: int = 512) -> str:
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
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=1,
            )
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)