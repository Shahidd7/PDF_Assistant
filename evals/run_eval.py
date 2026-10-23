import argparse
import json
import time
from pathlib import Path

import groq

import rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--questions", default="evals/questions.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fetch-k", type=int, default=20)
    ap.add_argument("--lambda-mult", type=float, default=0.5)
    ap.add_argument("--embed", default=rag.DEFAULT_EMBED)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    assert args.fetch_k >= args.k, "fetch_k must be >= k"

    text = Path(args.questions).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        questions = json.loads(text)
    else:
        questions = [json.loads(line) for line in text.splitlines() if line.strip()]

    t0 = time.perf_counter()
    _, chunks = rag.load_chunks(args.pdf, args.chunk_size, args.overlap)
    store = rag.build_store(chunks, args.embed)
    index_s = time.perf_counter() - t0
    print(f"Indexed {len(chunks)} chunks in {index_s:.1f}s\n")

    hits, rr_total, answerable = 0, 0.0, 0
    refused, unanswerable, errors = 0, 0, 0
    in_tokens = []

    for q in questions:
        if q["type"] == "answerable" and not q["pages"]:
            raise SystemExit(f"{q['id']}: fill in 'pages' before running.")
    for q in questions:
        docs = rag.retrieve(store, q["question"], args.k, args.fetch_k, args.lambda_mult)
        pages = [d.metadata["page"] for d in docs]

        if q["pages"]:
            answerable += 1
            rank = next((i for i, p in enumerate(pages, 1) if p in q["pages"]), None)
            if rank:
                hits += 1
                rr_total += 1 / rank
            print(f"{'HIT ' if rank else 'MISS'} {q['id']} expected={q['pages']} got={pages}")
        else:
            unanswerable += 1
            try:
                out = rag.generate(q["question"], docs)
            except groq.APIError as e:
                errors += 1
                print(f"ERR  {q['id']} {type(e).__name__}")
                continue
            ok = rag.REFUSAL.lower() in out["answer"].lower()
            refused += ok
            if out["input_tokens"]:
                in_tokens.append(out["input_tokens"])
            print(f"{'REFUSED' if ok else 'ANSWERED (bad)'} {q['id']}: {out['answer'][:80]!r}")
            time.sleep(2)   # be gentle with free-tier rate limits

    summary = {
        "label": args.label,
        "k": args.k, "fetch_k": args.fetch_k, "lambda_mult": args.lambda_mult,
        "embed": args.embed, "chunk_size": args.chunk_size, "overlap": args.overlap,
        "chunks": len(chunks), "index_s": round(index_s, 1),
        "recall_at_k": f"{hits}/{answerable}",
        "mrr": round(rr_total / answerable, 3) if answerable else None,
        "refusal_correct": f"{refused}/{unanswerable - errors}",
        "avg_input_tokens": round(sum(in_tokens) / len(in_tokens)) if in_tokens else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    with open("evals/results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

if __name__ == "__main__":
    main()