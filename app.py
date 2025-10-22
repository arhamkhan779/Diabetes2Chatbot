import streamlit as st
from streamlit_chat import message
import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
import logging
load_dotenv()

api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

faiss_index = faiss.read_index(r"faq_index.faiss")
with open(r"faq_texts.pkl", "rb") as f:
        text_chunks = pickle.load(f)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

template = """
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

def generate_embeddings(text: str):
    """Generate embeddings using Gemini embedding model (synchronous)"""
    try:
        response =  genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        if response and "embedding" in response:
            return response["embedding"]
        else:
            logging.warning("⚠️ No embedding returned for text.")
            return []
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return []


def find_similar_texts(query: str, top_k=5):
    """Find similar chunks using FAISS"""
    query_embedding = np.array(generate_embeddings(query)).astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [text_chunks[idx].page_content for idx in indices[0] if idx < len(text_chunks)]




def generate_response(query: str, name: str):
    """Generate natural language response based on query and context (synchronous)"""
    try:
        context =  find_similar_texts(query)
        formatted_prompt = template.format(context=context, query=query, name=name)
        # Use the synchronous generate_content method
        response =  model.generate_content(formatted_prompt)

        if not response or not getattr(response, "text", None):
            return "⚠️ Sorry, I couldn’t generate a valid response. Please try again."

        return response.text
    except Exception as e:
        logging.error(f"Response generation failed: {e}")
        return f"⚠️ An internal error occurred while generating the response: {e}"
    




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