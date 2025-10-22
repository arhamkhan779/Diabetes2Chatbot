import streamlit as st
from streamlit_chat import message
from backend import generate_response
st.set_page_config(page_title="DiaGuide Assistant", page_icon="💉", layout="wide")

st.title("💉 DiaGuide Assistant")
st.subheader("AI-powered support for **Type 2 Diabetes Mellitus Guidelines**")


st.sidebar.header("⚙️ Settings")
user_name = st.sidebar.text_input("Your Name", value="User")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


query = st.chat_input("Ask about Type 2 Diabetes Mellitus management...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.spinner("Analyzing guidelines..."):
        response = generate_response(query, user_name)
    st.session_state["messages"].append({"role": "assistant", "content": response})


for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=msg["content"])
    else:
        message(msg["content"], is_user=False, key=msg["content"])