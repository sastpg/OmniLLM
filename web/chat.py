import json
import streamlit as st
from tools import TOOL_DESCRIPTIONS, despatch_tool
from api import list_llm_models, call_llm, get_default_idx

st.set_page_config(layout="wide", page_title='OmniLLM')
st.title("OmniLLM") # 开源大模型应用平台
st.write("""OmniLLM 是一个开源大模型应用平台，支持多种开源大模型的在线应用如对话、微调、对齐等。代码仓库：[OmniLLM](https://github.com/sastpg/OmniLLM)""")

# cols = st.columns(4)
# cols[0].container(height=120).write(':balloon:')
# cols[1].container(height=120).write(':balloon:')
# cols[2].container(height=120).write(':balloon:')
# cols[3].container(height=120).write(':balloon:')

SYSTEM_PROMPT = "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"

with st.sidebar:
    models = list_llm_models()
    model_name = st.selectbox('选择模型', models, index=get_default_idx(models, "glm-4-9b-chat"))
    system_prompt = st.text_area('系统提示词', value=SYSTEM_PROMPT, height=100, help="点击下面的“Apply”按钮更改生效")
    cols = st.columns(2)
    apply = cols[0].button("Apply", use_container_width=True, help="应用系统提示词")
    clear_history = cols[1].button("Clear", use_container_width=True, help="清空对话历史")
    stream = cols[0].checkbox(label="流式输出", value=False)
    do_remember = cols[1].checkbox(label="多轮对话", value=False)
    use_tool = cols[0].checkbox(label="使用工具", value=True)
    if use_tool:
        tool_list = st.multiselect(
            "可选工具",
            ["web_search", "get_weather", "draw", "code_executor", "ppt_maker"],
            ["web_search", "draw"]
        )
    with st.expander("模型配置", expanded=False):
        top_p = st.slider("top_p", 0.0, 1.0, 0.9, step=0.01)
        top_k = st.slider("top_k", 1, 50, 50, step=1, key="top_k")
        temperature = st.slider("temperature", 0.0, 1.0, 0.7, step=0.01)
        # repetition_penalty = st.slider("repetition_penalty", 0.0, 2.0, 1.0, step=0.01)
        max_new_tokens = st.slider("max_new_tokens", 1, 4096, 512, step=1)

    
if "messages" not in st.session_state:
    st.session_state["messages"] = [dict(role="system", content=SYSTEM_PROMPT)]

if apply or clear_history:
    st.session_state.messages = [dict(role="system", content=system_prompt)] if system_prompt!="" else []

for msg in st.session_state.messages:
    if msg["role"] == "system":
        pass
    elif msg["role"] == "tool":
        st.chat_message(msg["role"]).expander("Observation", expanded=False).write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Chat with OmniLLM!"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    print(st.session_state.messages)
    # tool encapsulation
    tools = []
    if use_tool:
        for tool in tool_list:
            tools.append(dict(name=tool, description=TOOL_DESCRIPTIONS[tool]))
    
    for _ in range(6):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            tool_placeholder = st.empty()
        for resp in call_llm(
            model_name,
            st.session_state["messages"],
            tools=tools, stream=stream,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            max_new_tokens=max_new_tokens
        ):
            content = resp["content"]
            if content:
                message_placeholder.markdown(content + "●")
        else:
            if content:
                message_placeholder.markdown(content)
            st.session_state.messages.append(resp)
            tool_calls = resp["tool_calls"]
            if tool_calls:
                with st.spinner(f"Calling tool {tool_calls['name']}..."):
                    observation = despatch_tool(tool_calls['name'], json.loads(tool_calls['arguments']))
                tool_placeholder.expander("Observation", expanded=False).write(observation.text)
                if observation.content_type == "image":
                    st.chat_message("assistant").image(observation.image_url)
                st.session_state.messages.append({"role": "tool", "content": observation.text})
            else:
                break