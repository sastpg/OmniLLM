from utils import list_llm_models, call_llm, get_default_idx
import streamlit as st

st.set_page_config(layout="wide", page_title='OmniLLM')
st.title("OmniLLM Demo")


with st.sidebar:
    models = list_llm_models()
    model = st.selectbox('选择模型类型', models, index=get_default_idx(models, "glm-4-9b-chat"))
    top_p = st.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    top_k = st.slider("top_k", 1, 20, 10, step=1, key="top_k")
    temperature = st.slider("temperature", 0.0, 1.5, 0.95, step=0.01)
    repetition_penalty = st.slider("repetition_penalty", 0.0, 2.0, 1.0, step=0.01)
    max_new_tokens = st.slider("max_new_tokens", 1, 4096, 2048, step=1)
    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    resp = call_llm(st.session_state["messages"], top_p=0.9, temperature=0.7, do_sample=True, max_new_tokens=512)
    response = resp["data"].strip()
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)