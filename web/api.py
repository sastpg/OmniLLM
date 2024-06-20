import requests

def call_llm(model_name, messages, **kwargs):
    url = f"http://localhost:8001/{model_name}"
    message = {
        "messages": messages,
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