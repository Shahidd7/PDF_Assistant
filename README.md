# Ask Your PDF: a RAG question-answering app with a measured evaluation

Upload a text-based PDF and ask questions. Answers come **only** from the document, cite
page numbers, and the app refuses when the document doesn't cover the question. The
retrieval quality is measured with a small evaluation harness rather than assumed.

**Demo:** [link or screen recording] | **Stack:** FastAPI, Chroma, sentence-transformers, Groq, vanilla JS

![screenshot](docs/screenshot.png)

## What it does
- Upload a PDF (limits: 50 MB, 400 pages, text-based only), then chat with it.
- Every answer shows the retrieved pages and expandable source snippets, so a user can check the citations.
- Out-of-scope questions return a fixed refusal sentence instead of a guess.
- Each upload gets its own in-memory index, so documents never mix between uploads.

## Architecture
```
Browser (HTML/CSS/JS)
   │  POST /api/upload (PDF)        POST /api/ask (doc_id + question)
   ▼
FastAPI (api.py)  ── validation, error mapping, per-request logging
   ▼
rag.py
   ├─ load_chunks: PyPDF → 1000-char chunks, 150 overlap, 1-based page metadata
   ├─ build_store: MiniLM embeddings → in-memory Chroma collection (one per upload)
   ├─ retrieve: MMR search, k=5 of 20 candidates
   └─ generate: Groq LLM, temperature 0, answer only from context
```

## Design decisions and why
Settings chosen with evidence are marked **measured**. Others are marked **not tuned**, and I say so.

| Decision | Choice | Reason |
|---|---|---|
| Top-k | 5 (**measured**) | k=8 gave the same recall (18/18) and MRR (0.883 vs 0.886) but about 50% more input tokens (1534 vs 1022). On a free tier, tokens are the limit. |
| Embedding model | all-MiniLM-L6-v2 (**measured**) | bge-small-en-v1.5 scored lower on my set (recall 16/18, MRR 0.775 vs 0.886) and indexed about twice as slowly. I did not test bge's optional query prefix, so this means "did not beat MiniLM in my setup", not "bge is worse". |
| Retrieval | MMR (**not tuned**) | Books repeat ideas across chapters; MMR avoids returning near-duplicate chunks. The `lambda_mult=0.5` value was not tuned on the final question set. |
| Chunk size | 1000 chars, 150 overlap (**not tuned**) | A starting point: roughly a few paragraphs, within MiniLM's ~256-token input limit. Chunk-size experiments are listed under Future work. |
| LLM | `openai/gpt-oss-20b` on Groq | Free tier and fast. Not benchmarked against other models. The model name lives in `.env` because Groq retires models. |
| Temperature | 0 | Repeatable behavior, which evaluation needs. Small variation can remain. |
| Per-upload in-memory index | Chroma, no persistence | Prevents one user's questions retrieving another user's document. Cost: the index disappears on restart. |
| Prompt | "Answer ONLY from context", exact refusal sentence, no "implied" points | An earlier answer padded in points the book only implied; the prompt now forbids it. |

## Evaluation
**Method.** 22 questions I wrote myself: 18 answerable with the correct PDF pages labeled, and 4 unanswerable.
Labels were found with plain text search (`evals/find_pages.py`), independently of the retriever.
The harness (`evals/run_eval.py`) measures retrieval without calling the LLM, then checks refusals with the LLM.
Test document: *Designing Machine Learning Systems* (a copyrighted book, so it is not included in this repo).

- **Recall@k:** did any retrieved chunk come from a correct page?
- **MRR:** how early did the first correct page appear?
- **Refusal:** did the bot refuse the unanswerable questions?

| Run | Change | Recall | MRR | Avg input tokens | Index time |
|---|---|---|---|---|---|
| baseline2 | k=5, MiniLM | 18/18 | 0.886 | 1022 | ~110 s |
| k8 | k=8 | 18/18 | 0.883 | 1534 | ~117 s |
| bge-small | bge-small, k=5 | 16/18 | 0.775 | 1048 | ~218 s |

Refusals were correct on 4/4 in all runs. Index times were measured on a dual-core laptop CPU and varied between runs, so treat them as rough.

**What the baseline gets wrong.** Recall is saturated, so MRR is the useful number. The correct page ranked first for 15 of 18
questions. The misses: for q10 (shadow vs canary) the table of contents ranked first; for q19 (few labels) the correct page was 4th;
for q18 (detecting shift, phrased in plain words while the book says "two-sample test") it was 5th.

**A bug in my own evaluation.** My first question set scored 0/9. Two different embedding models failed identically, which pointed at the test rather than the models.
My page labels used the book's printed page numbers, which were 20 behind the PDF's page positions. I confirmed the offset against the book's own index, corrected the labels, and the baseline went to 12/12.
That was too easy (wide labels, questions copied from the book's wording), so I tightened labels and added paraphrased questions.

**Limits of this evaluation.** 18 answerable questions is small: one question is about 5.5 points. Differences of one question are noise.
Some labels cover several pages, which makes a hit easier. I labeled the questions myself.
The eval measures retrieval and refusal, not the quality of the generated answers.

## Reliability and security
- Upload checks: `.pdf` extension, `%PDF` signature, 50 MB size cap, 400 page cap, and an error for scanned PDFs with no text.
- Model output is rendered with `marked` and sanitized with DOMPurify, so injected HTML or scripts in a PDF or an answer can't run.
- Provider errors (rate limit, connection, other) become clear 429, 503 and 502 responses, and the UI stays usable.
- Each request logs retrieval time, LLM time, and input/output token counts. Question text is not logged.
- The bot has no tools, so a malicious PDF can't make it act. The prompt reduces, but does not eliminate, prompt-injection risk.
- Uploads are sent to a third-party free API. Don't upload sensitive documents.
- Tests: `pytest` covers validation, error mapping and the refusal contract with retrieval and the LLM mocked.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env      # set GROQ_API_KEY and GROQ_MODEL
python -m uvicorn api:app
```
Open http://127.0.0.1:8000. Interactive API docs at `/docs`.

**Docker:** `docker build -t pdf-qa .` then `docker run --env-file .env -p 8000:8000 pdf-qa`

**Tests:** `python -m pytest -v`
**Evaluation:** `python -m evals.run_eval --pdf your.pdf --label baseline --questions evals/questions.json`

## API
- `POST /api/upload` (multipart `file`) → `{doc_id, pages, chunks}`
- `POST /api/ask` (`{doc_id, question}`) → `{answer, pages, sources}`

## Limitations
- Text-based PDFs only (no OCR). Tables and multi-column layouts may extract poorly.
- In-memory index: lost on restart, no user accounts, no rate limiting, not hardened for public hosting.
- No conversation memory: each question is answered on its own, so follow-ups like "what about its cost?" can fail.
- The headline evaluation used a copyrighted book that I can't share, so others can't reproduce those exact numbers.
- Evaluation set is small and self-labeled.

## Future work
- Skip the table of contents and index when indexing (they took retrieval slots in several questions).
- Test a cross-encoder reranker against the MRR failures above.
- Tune chunk size and overlap with the same harness.
- Query rewriting for follow-up questions.
- Re-run the evaluation on a freely licensed document so the results are reproducible.
- Background indexing with a progress bar for large uploads.

## What I learned
Write your own 3-4 sentences here: the offset bug, MRR vs recall, and what you'd do differently.