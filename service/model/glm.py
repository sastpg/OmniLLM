import torch
from pathlib import Path

from typing import Union, List, Dict
from model.common import AbstractModel, StopAtTokens, preprocess_message, postprocess_message
from transformers.generation.stopping_criteria import StoppingCriteriaList
from utils.logger import logger

class GLM(AbstractModel):
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        super().__init__(model_path, device_map)

    def __call__(self, messages: List[Dict], tools: List[Dict]=None, stream=False, **kwargs) -> str:
        if tools:
            messages = preprocess_message(messages, tools)
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(StopAtTokens(token_id_list=self.tokenizer.encode('OBSERVATION')[2:]))
            kwargs['stopping_criteria'] = stopping_criteria
        
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
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(response)
            return postprocess_message(response)