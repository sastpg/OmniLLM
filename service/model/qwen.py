from pathlib import Path

from typing import Union, List,Dict
from model.common import AbstractModel

class Qwen(AbstractModel):
    def __init__(self, model_path: Path, device_map: Dict | str = '') -> None:
        super().__init__(model_path, device_map)

    def __call__(self, messages: List[Dict[str, str]], temperature: float=0.6, top_p: float=0.9, do_sample: bool=True, max_new_tokens: int=512) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Directly use generate() and tokenizer.decode() to get the output.
        # Use `max_new_tokens` to control the maximum output length.
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response