# Resume ↔ Job Matcher (RAG)

Upload a resume and get the best-matching job postings, the skills you already cover, the skills you're missing, and concrete suggestions, with every claim backed by a quote.

## How it works

```
resume (PDF/DOCX/TXT)
   │  parse → redact email/phone/links → split into sections
   ▼
LLM extracts skills + evidence quotes ──► quotes verified against the resume (hallucinations dropped)
   │
   ▼
embed resume sections (fastembed, in memory) ──► search prebuilt FAISS index of job postings (top 20)
   │
   ▼
hybrid score = 0.6 × embedding similarity + 0.4 × skill overlap ──► top 5
   │
   ▼
LLM gap analysis per match (grounded in the retrieved job text) ──► quotes verified again
```

The job index is built offline and shipped with the repo. Resumes are never stored or logged.

## Embeddings and vector database

This project uses **FastEmbed** to generate dense semantic embeddings and **FAISS**
as its persistent vector store:

- Job title + description text is embedded when `scripts/build_index.py` runs.
- Vectors are L2-normalized and stored in a FAISS `IndexFlatIP` index. Inner-product
  search on normalized vectors is equivalent to cosine similarity.
- Job metadata (ID, title, company, description, skills and role) is stored in
  `index/jobs_meta.json` and mapped to FAISS vector IDs by row order.
- `index/index_manifest.json` records the embedding model, vector dimension,
  vector count and similarity configuration. The retriever checks this manifest
  to detect stale or mismatched indexes.
- At request time, the uploaded resume is redacted, split into sections and
  embedded in memory. Its vectors query FAISS; resume vectors are not persisted.
- The top semantic candidates are reranked using the hybrid embedding + skill score.
- Each result exposes the raw cosine similarity (embedding score), cosine distance
  (`1 - similarity`) and Euclidean distance (`sqrt(2 - 2 × similarity)`). Because
  stored and query vectors are L2-normalized, these distances are comparable for
  the same embedding model. Lower distance means the resume chunk and job vector
  are closer; higher cosine similarity means a stronger semantic match.
- Expand **Embedding & vector distance details** on each result card to inspect
  these values. They are measured between the resume section vector that produced
  the best match and that job's vector in FAISS, not between every pair of jobs.

Rebuild the index whenever job data or `EMBED_MODEL` changes:

```bash
python scripts/build_index.py --csv data/jobs_sample.csv --max-jobs 5000
```

FAISS is a local, file-backed vector index rather than a hosted database service.
This keeps deployment simple and avoids a separate database bill.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env            # then add GEMINI_API_KEY (or GROQ_API_KEY + LLM_PROVIDER=groq)

python scripts/make_sample_data.py     # synthetic demo postings (replace with real data later)
python scripts/build_index.py          # downloads the embedding model, writes index/
uvicorn app.main:app --reload          # open http://127.0.0.1:8000
```

Without an API key the app still works, using keyword matching instead of the LLM.

## Use real job data

1. Download a job-postings CSV (for example a Kaggle "LinkedIn job postings" dataset) into `data/raw/`.
2. Make sure it has `title`, `company`, `description` columns (rename them if needed). Optional: `skills`, `role`.
3. `python scripts/build_index.py --csv data/raw/postings.csv --max-jobs 5000`
4. Commit the new `index/` folder. Keep it to a few thousand postings so it fits Render's free tier.

## Evaluate (this is what makes the project stand out)

```bash
python scripts/evaluate.py --k 5
```

Compares a TF-IDF baseline, embeddings only, and the hybrid score using Recall@k, Precision@1 and MRR.
The included `eval/` set has only 5 resumes against synthetic postings, so it is too easy to prove anything.
Add 20-30 real resumes (anonymised) to `eval/resumes/`, label them in `eval/labels.json`, and put your results table here.
Use the evaluation to tune `W_EMBED` / `W_SKILL` instead of guessing.

## Run the tests

```bash
pytest -q
```

The tests run offline using a stand-in embedder, so no model download or API key is needed.

## Deploy to Render

1. Push this repo to GitHub (the `index/` folder must be included; `.env` must not be).
2. In Render choose **New → Blueprint** and select the repo (it reads `render.yaml`), or create a **Web Service** manually with:
   - Build command: `pip install -r requirements.txt && python scripts/warm_model.py`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Add `GEMINI_API_KEY` (or `GROQ_API_KEY`) under **Environment**.
4. Open the URL and test with a real resume.

Free tier notes: the service sleeps after about 15 minutes idle (first request takes 30-60 s), RAM is 512 MB, and the disk is wiped on every deploy, which is why the index is committed instead of generated at runtime.
If you see out-of-memory crashes, shrink the index (`--max-jobs`) or switch to an API-based embedding model.

## Configuration

All settings are environment variables (see `app/config.py`): `LLM_PROVIDER`, `LLM_MODEL`, `EMBED_MODEL`, `TOP_K_RETRIEVE`, `TOP_N_RESULTS`, `ANALYZE_TOP`, `W_EMBED`, `W_SKILL`, `SIM_FLOOR`, `SIM_CEIL`, `MAX_UPLOAD_MB`, `RATE_LIMIT_PER_HOUR`.

- LLM model names change over time. If a call fails with a 404, check your provider's current model list and set `LLM_MODEL`.
- If you change `EMBED_MODEL`, rebuild the index and re-tune `SIM_FLOOR` / `SIM_CEIL`, which map cosine similarity to the 0-100 score.

## Project layout

```
app/        main.py (API) · parser.py · extractor.py · retriever.py · scorer.py · analyzer.py
            skills.py (vocabulary) · llm.py · embedding.py · indexing.py · config.py
static/     index.html (front end)
scripts/    build_index.py · evaluate.py · make_sample_data.py · warm_model.py
eval/       labelled resumes + labels.json
tests/      offline test suite
```

## Known limitations

- The skills vocabulary (`app/skills.py`) is small. Extend it for your target roles; the LLM can find skills outside it, but scoring only credits skills the vocabulary knows.
- Scanned or image-only PDFs can't be read (no OCR). Heavily designed multi-column PDFs may parse out of order.
- English only with the default embedding model. For Tamil/Hindi, switch to a multilingual model such as `intfloat/multilingual-e5-small` and rebuild the index.
- Match scores are a ranking aid, not a prediction of hiring outcomes.
