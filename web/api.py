import json
import requests
from tools import TOOL_DESCRIPTIONS

def call_llm(model_name, messages, tools=None, stream=False, **kwargs):
    url = f"http://localhost:8001/{model_name}"
    message = {
        "messages": messages,
        "tools": tools,
        "stream": stream,
        "llm_config": kwargs
    }
    print(model_name, message)
    response = requests.post(url, stream=stream, json=message)
    for content in response.iter_content(chunk_size=None):
        resp = content.decode("utf-8")
        resp = json.loads(resp)
        yield resp

def list_llm_models():
    url = "http://localhost:8001/models"
    resp = requests.get(url)
    return resp.json()['data']

def get_default_idx(options, default_value):
    try:
        return options.index(default_value)
    except Exception as e:
        return 0


if __name__ == '__main__':
    msg = [
        {"role": "user", "content": "How is the weather in Hangzhou today?"}
    ]

    # test tool_call
    tools = []
    tool_list = ['get_weather', 'web_search']
    for tool in tool_list:
        tools.append(dict(name=tool, description=TOOL_DESCRIPTIONS[tool]))

    for resp in call_llm(model_name="llama-3-8b-instruct", messages=msg, tools=tools, stream=False):
        print(resp)

