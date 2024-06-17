from utils import list_llm_models, call_llm, get_default_idx
import streamlit as st

st.set_page_config(layout="wide", page_title='OmniLLM')
st.title("OmniLLM Demo")


with st.sidebar:
    models = list_llm_models()
    model_name = st.selectbox('Select models', models, index=get_default_idx(models, "glm-4-9b-chat"))
    system_prompt = st.text_area('System prompt', value="你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。", height=100)
    cols = st.columns(2)
    apply = cols[0].button("Apply", use_container_width=True)
    reset = cols[1].button("Reset", use_container_width=True)
    stream = cols[0].checkbox(label="流式输出", value=False)
    do_remember = cols[1].checkbox(label="多轮对话", value=False)
    top_p = st.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    top_k = st.slider("top_k", 1, 20, 10, step=1, key="top_k")
    temperature = st.slider("temperature", 0.0, 1.5, 0.95, step=0.01)
    # repetition_penalty = st.slider("repetition_penalty", 0.0, 2.0, 1.0, step=0.01)
    max_new_tokens = st.slider("max_new_tokens", 1, 4096, 512, step=1)
    cols = st.columns(2)
    retry = cols[0].button("Retry", use_container_width=True)
    clear_history = cols[1].button("Clear", use_container_width=True)
    

if "messages" not in st.session_state:
    print("not in")
    st.session_state["messages"] = []

if clear_history:
    st.session_state.messages = []


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    resp = call_llm(model_name, system_prompt, st.session_state["messages"], top_p=top_p, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    response = resp["data"].strip()
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)