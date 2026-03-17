from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = TextLoader("sustainability_docs.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="vector_db"
)

vector_db.persist()

print("Vector database created")
