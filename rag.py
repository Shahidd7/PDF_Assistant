import os
import time
import uuid
from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
REFUSAL = "I could not find this in the document."

prompt = ChatPromptTemplate([
    ("system", "Answer ONLY from the context. Cite pages like [p. 12]. "
     "Be concise: a one-sentence answer, then short bullet points. No tables. "
     "Include only points the context states directly. Never add points that are "
     "implied. If the context covers only part of the question, say so. "
     f'If it is not covered at all, reply exactly: "{REFUSAL}"'),
     ("human" , "Context:\n{context}\n\nQuestion: {question}")
])

@lru_cache(maxsize = 2)
def get_embedding(model_name : str):
    return HuggingFaceEmbeddings(model_name = model_name)

@lru_cache(maxsize = 1)
def get_llm():
    return ChatGroq(
        model_name = os.environ["GROQ_MODEL"],
        temperature = 0,
        max_tokens = 1024,
        max_retries = 3,
    )

def load_chunks(pdf_path: str, chunk_size: int = 1000, overlap :int = 150 ):
    """returns (number of pages in the file, list of chunks with 1-based pages)."""
    pages = PyPDFLoader(pdf_path).load()
    n_pages = len(pages)
    pages = [p for p in pages if p.page_content.strip()]
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,chunk_overlap = overlap)
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata['page'] = c.metadata['page'] + 1
    return n_pages,chunks

def build_store(chunks, embed_model: str = DEFAULT_EMBED) -> Chroma:
    return Chroma.from_documents(
        documents = chunks,
        embedding = get_embedding(embed_model),
        collection_name = f"doc_{uuid.uuid4().hex}",
    )


def retrieve(store: Chroma, question: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5):
    return store.max_marginal_relevance_search(
        question, k = k, fetch_k = fetch_k, lambda_mult = lambda_mult
    )

def generate(question: str, docs) -> dict:
    context = "\n\n".join(f"[Page {d.metadata['page']}]\n{d.page_content}" for d in docs)
    start = time.perf_counter()
    response = get_llm().invoke(prompt.invoke({"context":context, "question": question}))
    usage = response.usage_metadata or {}
    return{
        "answer" : response.content,
        "latency_s" : round(time.perf_counter() - start, 2),
        "input_tokens" : usage.get("input_tokens"),
        "output_tokens" : usage.get("output_tokens"),
    }