from pathlib import Path

from typing import Union, List,Dict
from model.common import AbstractModel, StopAtTokens
from transformers.generation.stopping_criteria import StoppingCriteriaList

class Llama3(AbstractModel):
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        super().__init__(model_path, device_map)
    
    def __call__(self, messages: List[Dict[str, str]], tools: list=None, stream=False, **kwargs) -> str:
        if tools:
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(StopAtTokens(token_id_list=self.tokenizer.encode('OBSERVATION')[1:]))
            kwargs['stopping_criteria'] = stopping_criteria
        
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
            eos_token_id=terminators,
            **kwargs
        )
        if len(outputs) > 0:
            response = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True)
        else:
            return None

