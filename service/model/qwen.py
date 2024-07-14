from pathlib import Path
from typing import Union, List, Dict
from fastapi.responses import StreamingResponse
from model.common import AbstractModel, StopAtTokens, preprocess_message, postprocess_message
from transformers.generation.stopping_criteria import StoppingCriteriaList
from utils.logger import logger

class Qwen(AbstractModel):
    def __init__(self, model_path: Path, device_map: Union[dict,str]='') -> None:
        super().__init__(model_path, device_map)

    def __call__(self, messages: List[Dict], tools: List[Dict]=None, stream: bool=False, **kwargs) -> Union[Dict, StreamingResponse]:
        if tools and tools!=[{}]:
            messages = preprocess_message(messages, tools)
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(StopAtTokens(token_id_list=self.tokenizer.encode('OBSERVATION')))
            kwargs['stopping_criteria'] = stopping_criteria
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        if stream:
            gen_config = dict(model_inputs, streamer=self.streamer, **kwargs)
            return StreamingResponse(self.gen_stream(gen_config), media_type="application/x-ndjson")
            # print(generated_text)
        else:
            # Directly use generate() and tokenizer.decode() to get the output.
            # Use `max_new_tokens` to control the maximum output length.
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                **kwargs
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(response)
            return postprocess_message(response)