import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load env
load_dotenv()
api_key = st.secrets("GEMINI_API_KEY")

# Embedding & LLM
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=api_key
)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", api_key=api_key)

# Prompt template (adapted for Diabetes Mellitus 2 guidelines)
prompt_template = PromptTemplate(
    input_variables=["context", "query", "name"],
    template="""
    You are **DiaGuide Assistant**, a professional AI assistant specialized in **Type 2 Diabetes Mellitus guidelines**.  
    Your purpose is to provide **accurate, evidence-based, and context-aware guidance** to healthcare professionals, patients, and caregivers.  

    ### Rules:
    1. Base your answers strictly on the retrieved context from clinical guidelines.  
    2. If the context is insufficient, clearly state:  
       "_I don’t have enough information from the guidelines to answer this fully. Please refer to official diabetes care recommendations._"  
    3. Use clear, professional, and empathetic language.  
    4. Break down explanations into short, structured points with clarity.  
    5. Use medically relevant emojis (🩺🍎💉) sparingly for readability.  
    6. Mention the **user's name** in the response naturally.  

    ---
    ### Context:  
    {context}  

    ### Query:  
    {query}  

    ### User:  
    {name}  

    ---
    ### Response:
    """
)

# Load Vector Store
@st.cache_resource
def load_vector_store():
    return FAISS.load_local(
        "vectorestore",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

vector_store = load_vector_store()
retriever = vector_store.as_retriever()

def get_response(query, name):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = prompt_template.format(context=context, query=query, name=name)
    response = model.invoke(prompt)
    return response.content

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="DiaGuide Assistant", page_icon="💉", layout="wide")

st.title("💉 DiaGuide Assistant")
st.subheader("AI-powered support for **Type 2 Diabetes Mellitus Guidelines**")

# Sidebar
st.sidebar.header("⚙️ Settings")
user_name = st.sidebar.text_input("Your Name", value="User")

# Chat container
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
query = st.chat_input("Ask about Type 2 Diabetes Mellitus management...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.spinner("Analyzing guidelines..."):
        response = get_response(query, user_name)
    st.session_state["messages"].append({"role": "assistant", "content": response})

# Display chat
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=msg["content"])
    else:
        message(msg["content"], is_user=False, key=msg["content"])
