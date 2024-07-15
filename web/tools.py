import requests
from dataclasses import dataclass

@dataclass
class ToolObservation:
    content_type: str
    text: str
    image_url: str | None = None

def web_search(query) -> ToolObservation:
    subscription_key = "YOUR SUBSCRIPTION KEY"
    endpoint = "https://api.bing.microsoft.com/v7.0/search"

    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    content_list = []
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        contents = response.json()["webPages"]["value"][:6]
        for i, c in enumerate(contents):
            content_list.append(f"[{i+1}] {c['snippet']}")

        results = "\n".join(content_list)
        return ToolObservation(content_type="text", text=results)
    except Exception as ex:
        raise ex

def draw(description) -> ToolObservation:
    return ToolObservation(content_type="image", text="I have shown the image to users.", image_url="../images/api.png")

def get_weather(city_name) -> ToolObservation:
    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": [
            "temp_C",
            "FeelsLikeC",
            "humidity",
            "weatherDesc",
            "observation_time",
        ],
    }

    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback

        ret = (
                "Error encountered while fetching weather data!\n" + traceback.format_exc()
        )

    return ToolObservation(content_type="text", text=str(ret))


TOOL_DESCRIPTIONS = {
    "web_search": "web_search(query: str) -> str - Search the website to obtain relevant information.",
    "get_weather": "get_weather(city_name: str) -> str - Get the current weather for `city_name`.",
    "draw": "draw(description: str) -> str - Generate an image based on the description. The description should be described in detail, within 30 English words.",
}

ALL_TOOLS = {
    "web_search": web_search,
    "get_weather": get_weather,
    "draw": draw,
}

def despatch_tool(tool_name: str, args: dict) -> ToolObservation:
    return ALL_TOOLS[tool_name](**args)


if __name__ == "__main__":
    observation = despatch_tool("get_weather", {"city_name": "New York"})
    print(observation.content_type, observation.text, observation.image_url)