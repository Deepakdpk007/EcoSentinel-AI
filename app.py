import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "vector_db"
COLLECTION_NAME = "sustainability"

st.title("🌱 EcoSentinel AI")

# -------- LOAD EMBEDDING MODEL --------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- SAFE LOAD VECTOR DB --------
def load_vector_db():
    if not os.path.exists(PERSIST_DIR):
        st.error("Vector DB not found. Run rag_setup.py first.")
        return None
    
    try:
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
        return db
    
    except Exception as e:
        st.error("⚠️ Vector DB corrupted or incompatible. Rebuilding required.")
        st.code(str(e))
        return None


vector_db = load_vector_db()

# -------- QUERY --------
query = st.text_input("Ask sustainability recommendation:")

if query and vector_db:
    docs = vector_db.similarity_search(query, k=3)

    st.subheader("📊 Recommendations:")
    for doc in docs:
        st.write("-", doc.page_content)
