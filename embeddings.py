import os
import logging
import google.generativeai as genai
import faiss
import numpy as np
import pickle
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Securely configure Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")  # Store API key in environment variables
if not API_KEY:
    logging.error("API key not found. Set GEMINI_API_KEY as an environment variable.")
    exit(1)
genai.configure(api_key=API_KEY)

# Step 1: Load data from text file
logging.info("Loading Text Document")


# Step 2: Load data from PDFs
pdf_files = [
    "type-2-diabetes-in-adults-management-pdf-1837338615493.pdf"
]
pdf_documents = []
for pdf in pdf_files:
    logging.info(f"Loading PDF Document: {pdf}")
    try:
        pdf_loader = PyPDFLoader(pdf)
        pdf_documents.extend(pdf_loader.load())
    except Exception as e:
        logging.error(f"Error loading {pdf}: {e}")

# Step 3: Merge all documents
logging.info("Concatenating all documents")
all_documents = pdf_documents

# Step 4: Split text into chunks
logging.info("Splitting documents into chunks")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = splitter.split_documents(all_documents)

# Step 5: Function to generate embeddings using Gemini
logging.info("Initializing embedding generation")
def generate_embeddings(text):
    try:
        response = genai.embed_content(model="models/text-embedding-004", content=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None

# Step 6: Compute embeddings
logging.info("Generating embeddings for text chunks")
embeddings = [generate_embeddings(chunk.page_content) for chunk in text_chunks]
embeddings = [e for e in embeddings if e is not None]  # Remove failed embeddings

if not embeddings:
    logging.error("No embeddings generated. Exiting.")
    exit(1)

# Step 7: Convert embeddings to NumPy array
embeddings_np = np.vstack(embeddings)

# Step 8: Create FAISS index and add embeddings
logging.info("Creating FAISS index")
dim = embeddings_np.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings_np)


# Step 9: Save FAISS index and text chunks
logging.info("Saving FAISS index and text data")
faiss.write_index(faiss_index, "faq_index.faiss")
with open("faq_texts.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

logging.info("FAISS index built successfully!")
