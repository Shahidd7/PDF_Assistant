# Decisions log

## 1. Evaluation set v1 was wrong (scored 0/9)
- Symptom: recall 0/9 on MiniLM and 0/9 on bge-small, no setting helped.
- Reasoning: two different embedding models failing identically pointed at the test, not the models.
- Cause: my page labels used the book's printed page numbers, 20 behind the PDF positions.
- Verification: the book's own index lists shadow deployment on p. 282; the PDF has it at position 302.
- Fix: shifted labels by +20 and re-checked them against text search. Corrected baseline: 12/12.
- Lesson: validate the test before tuning the system.

## 2. Evaluation set v2 (22 questions: 18 answerable, 4 unanswerable)
- 12/12 was too easy: wide labels, and questions that reused the book's wording.
- Tightened labels to 1-4 pages, added paraphrased questions written without the book's terms.
- Labels were found by plain text search, independent of the retriever.

## 3. Experiments (same questions, one change per run)
| Run | Change | Recall | MRR | Avg input tokens | Decision |
|---|---|---|---|---|---|
| baseline2 | k=5, MiniLM | 18/18 | 0.886 | 1022 | baseline |
| k8 | k=8, fetch_k=30 | 18/18 | 0.883 | 1534 | rejected: no gain, ~50% more tokens |
| bge-small | bge-small, k=5 | 16/18 | 0.775 | 1048 | rejected: lower MRR, ~2x slower indexing |

Caveats: 18 answerable questions is small (one question is about 5.5 points); bge-small was tested without its optional query prefix;
index times varied between runs on a dual-core CPU.

## 4. Failure analysis of the baseline
- q10 (shadow vs canary): the table of contents (p. 10) ranked first, correct page second.
- q19 (few labels): correct page ranked 4th.
- q18 (detecting shift): correct page ranked 5th. The question used plain words; the book says "two-sample test".
- Pattern: front and back matter (contents, index) take retrieval slots without containing answers.

## 5. Prompt change
- An early answer included points the book only implied ("Implicit in the discussion of...").
- Prompt now forbids implied points and gives an exact refusal sentence.
- Result: refusals correct 4/4 across all runs. This tests the refusal path, not answer quality.

## 6. Not yet tuned
- Chunk size and overlap (1000 / 150), MMR lambda (0.5), the choice of LLM.

## 7. Docker
- First build failed at runtime: `langchain_chroma` was missing from requirements.txt
  (it was installed locally but not listed). Fixed by listing every direct import.
- fill: image size, and that you ran it and uploaded a PDF successfully.

## Next experiments
1. Skip front and back matter when indexing.
2. Add a cross-encoder reranker and compare MRR on q10, q18, q19.
3. Re-run the whole evaluation on a freely licensed document.