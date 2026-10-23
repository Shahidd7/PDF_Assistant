from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

data = PyPDFLoader("Designing_Machine Learning_Systems.pdf")
docs = data.load()

if os.path.exists("chroma_db"):
    print("Database already exists. Delete the folder to rebuild.")
    raise SystemExit

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150)

chunks = splitter.split_documents(docs)

for c in chunks:
    c.metadata["source"] = "designing_ml_systems"
    c.metadata["page"] = c.metadata["page"] + 1  

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    persist_directory = "chroma_db"
)