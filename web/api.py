import requests
from tools import TOOL_DESCRIPTIONS

def call_llm(model_name, messages, tools, **kwargs):
    url = f"http://localhost:8001/{model_name}"
    message = {
        "messages": messages,
        "tools": tools,
        "llm_config": kwargs
    }
    print(model_name, message)
    resp = requests.post(url, json=message)
    return resp.json()

def list_llm_models():
    url = "http://localhost:8001/models"
    resp = requests.get(url)
    return resp.json()['data']

def get_default_idx(options, default_value):
    try:
        return options.index(default_value)
    except Exception as e:
        return 0


if '__name__' == '__main__':
    msg = [
        {"role": "user", "content": "How is the weather in Hangzhou today?"}
    ]

    # test tool_call
    tools = []
    tool_list = ['get_weather', 'web_search']
    for tool in tool_list:
        tools.append(TOOL_DESCRIPTIONS[tool])
    print(tools)
    resp = call_llm(model_name="llama-3-8b-instruct", messages=msg, tools=tools)
    response = resp["data"].strip()
    print(response)