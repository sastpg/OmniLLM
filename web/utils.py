import requests

def call_llm(messages, **kwargs):
    url = "http://localhost:8001/llama-3-8b-instruct"
    message = {
        "messages": messages,
        **kwargs
    }
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