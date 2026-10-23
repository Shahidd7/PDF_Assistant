import logging
import os
import tempfile
import time
import uuid
from collections import OrderedDict

import groq
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag   

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pdfqa")

MAX_FILE_MB = 50
MAX_PAGES = 400
MAX_DOCS = 3
K = int(os.getenv("TOP_K", 5))
FETCH_K = int(os.getenv("FETCH_K", 20))
LAMBDA = float(os.getenv("LAMBDA_MULT", 0.5))

stores = OrderedDict()   # doc_id -> Chroma store

app = FastAPI(title="PDF Q&A")


class AskRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=500)


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are allowed.")

    limit = MAX_FILE_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_FILE_MB} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="This does not look like a real PDF.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        n_pages, chunks = rag.load_chunks(tmp_path)
    except Exception:
        log.exception("PDF read failed")
        raise HTTPException(status_code=422, detail="Could not read this PDF.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if n_pages > MAX_PAGES:
        raise HTTPException(status_code=413, detail=f"Too many pages (max {MAX_PAGES}).")
    if not chunks:
        raise HTTPException(status_code=422, detail="No text found. Scanned PDFs are not supported.")

    t0 = time.perf_counter()
    doc_id = uuid.uuid4().hex
    stores[doc_id] = rag.build_store(chunks)
    log.info("upload doc=%s pages=%d chunks=%d index_s=%.1f",
             doc_id[:8], n_pages, len(chunks), time.perf_counter() - t0)

    while len(stores) > MAX_DOCS:
        _, old = stores.popitem(last=False)
        old.delete_collection()

    return {"doc_id": doc_id, "pages": n_pages, "chunks": len(chunks)}


@app.post("/api/ask")
def ask(req: AskRequest):
    store = stores.get(req.doc_id)
    if store is None:
        raise HTTPException(status_code=404, detail="Document not found. Please upload the PDF again.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is empty.")

    t0 = time.perf_counter()
    docs = rag.retrieve(store, question, K, FETCH_K, LAMBDA)
    retrieve_s = time.perf_counter() - t0

    try:
        result = rag.generate(question, docs)
    except groq.RateLimitError:
        log.warning("ask doc=%s rate_limited", req.doc_id[:8])
        raise HTTPException(status_code=429, detail="Rate limit hit. Wait a few seconds.")
    except groq.APIConnectionError:
        log.warning("ask doc=%s connection_error", req.doc_id[:8])
        raise HTTPException(status_code=503, detail="Can't reach the model provider.")
    except groq.APIStatusError as e:
        log.warning("ask doc=%s api_error=%s", req.doc_id[:8], e.status_code)
        raise HTTPException(status_code=502, detail=f"Model API error {e.status_code}.")

    log.info("ask doc=%s k=%d retrieve_s=%.2f llm_s=%.2f in_tok=%s out_tok=%s",
             req.doc_id[:8], K, retrieve_s, result["latency_s"],
             result["input_tokens"], result["output_tokens"])

    return {
        "answer": result["answer"],
        "pages": sorted({d.metadata["page"] for d in docs}),
        "sources": [{"page": d.metadata["page"], "snippet": d.page_content[:250]} for d in docs],
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")