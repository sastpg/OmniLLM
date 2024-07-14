import json
import copy
import torch
from pathlib import Path
from threading import Thread
from utils.logger import logger
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Iterable
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings


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
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    @abstractmethod
    def __call__(self, messages: List[Dict], tools: List[Dict]=None, stream: bool=False, **kwargs):
        raise NotImplementedError

    def gen_stream(self, gen_config: Dict) -> Iterable[str]:
        # logger.info(gen_config)
        thread = Thread(target=self.model.generate, kwargs=gen_config)
        thread.start()

        generated_text = ""
        FN_TOKENS = ("\nACTION:", "\nARGS:", "\nOBSERVATION:", "\nRETURN:")
        for new_text in self.streamer:
            generated_text += new_text
            if not any(new_text in s for s in FN_TOKENS):
                response = postprocess_message(generated_text)
                yield json.dumps(response, ensure_ascii=False)
    
        logger.info(generated_text)


__PREFIX = "You are a helpful assistant."

__TOOL_INSTRUCTION = """You have access to the following tools:

{tools}

When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

ACTION: The tool to take, should be one of [{tool_names}]
ARGS: The input to the tool, should be a JSON blob with keys that match the tool's parameters names
OBSERVATION: Tool results
RETURN: Reply based on tool results."""


def preprocess_message(messages: List[Dict], tools: List[Dict]) -> List[Dict]:
    _messages = copy.deepcopy(messages)
    if _messages[0]["role"] != "system":
        _messages = [dict(role="system", content="")] + _messages
    chat_messages = []
    for msg in _messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            tool_names = ', '.join(tool.get('name') for tool in tools)
            tool_descs = '\n'.join(tool.get('description') for tool in tools)
            if not content:
                content = __PREFIX
            content += '\n\n' + __TOOL_INSTRUCTION.format(tools=tool_descs, tool_names=tool_names)
            chat_messages.append(dict(role=role, content=content))
        elif role == "user":
            chat_messages.append(msg)
        elif role == "assistant":
            # content = (content or [])
            tool_call = msg.get("tool_calls", None)
            if tool_call:
                tool_name, tool_args = tool_call["name"], tool_call["arguments"]
                content += f"\nACTION: {tool_name}\nARGS: {tool_args}"
            if chat_messages[-1]["role"] == "assistant":
                    chat_messages[-1]["content"] += content
            else:
                chat_messages.append(dict(role=role, content=content))
        elif role == "tool":
            chat_messages[-1]["content"] += f"\nOBSERVATION: {content}\nRETURN: "
        else:
            raise TypeError(f"Unexpected role: {role}")

    return chat_messages


def postprocess_message(messages: str) -> Dict:
    i = messages.find("ACTION:")
    # no tool call
    if i < 0:
        return {"role": "assistant", "content": messages, "tool_calls": None}
    messages = messages.rstrip('\nOBSERVATION')
    # content before tool call
    content = messages[:i].lstrip('\n').rstrip().rstrip('\n') if i > 0 else ''
    # tool call
    tool_info = messages[i:]
    i = tool_info.find("ARGS:")
    if i < 0:
        tool_name, tool_args=tool_info.split("ACTION:")[1], ""
    else:
        tool_name, tool_args=tool_info.split("ACTION:")[1].split("ARGS:")
    tool_name = tool_name.strip('\n').strip()
    tool_args = tool_args.strip('\n').strip()
    # print(tool_name, tool_args)
    tool_calls = {"name": tool_name, "arguments": tool_args}
    
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}
