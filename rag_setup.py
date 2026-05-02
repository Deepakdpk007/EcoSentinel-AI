import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# -------- SETTINGS --------
PERSIST_DIR = "vector_db"
COLLECTION_NAME = "sustainability"

# -------- STEP 1: CLEAN OLD DB --------
if os.path.exists(PERSIST_DIR):
    print("Deleting old vector DB...")
    shutil.rmtree(PERSIST_DIR)

# -------- STEP 2: LOAD DATA --------
with open("sustainability_docs.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

docs = [Document(page_content=chunk) for chunk in text_data.split("\n\n")]

# -------- STEP 3: EMBEDDINGS --------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- STEP 4: CREATE VECTOR DB --------
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME
)

vector_db.persist()

print("✅ Vector DB created successfully!")
