from pathlib import Path
from typing import Union, List, Dict
from fastapi.responses import StreamingResponse
from model.common import AbstractModel, StopAtTokens, preprocess_message, postprocess_message
from transformers.generation.stopping_criteria import StoppingCriteriaList
from utils.logger import logger

class Llama3(AbstractModel):
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        super().__init__(model_path, device_map)
    
    def __call__(self, messages: List[Dict], tools: List[Dict]=None, stream: bool=False, **kwargs) -> Union[Dict, StreamingResponse]:
        if tools and tools!=[{}]:
            messages = preprocess_message(messages, tools)
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
        
        if stream:
            gen_config = dict(input_ids=input_ids, streamer=self.streamer, eos_token_id=terminators, **kwargs)
            return StreamingResponse(self.gen_stream(gen_config), media_type="application/x-ndjson")
        else:
            outputs = self.model.generate(
                input_ids,
                eos_token_id=terminators,
                **kwargs
            )
            if len(outputs) > 0:
                generated_ids = outputs[0][input_ids.shape[-1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                logger.info(response)
                return postprocess_message(response)
            else:
                return None
        
