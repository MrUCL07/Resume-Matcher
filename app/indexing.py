"""Build the job-description index (run offline on your laptop, then commit the result)."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from .embedding import EmbedFn
from .skills import canonical, find_skills

TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    text = TAG_RE.sub(" ", text or "")
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_jobs(csv_path: str | Path, max_jobs: int | None = None) -> list[dict]:
    """Read a CSV with columns title, company, description (+ optional skills, role).

    Duplicates are removed. Skills come from the optional `skills` column plus
    anything the vocabulary finds in the title and description.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    jobs: list[dict] = []
    seen: set[tuple] = set()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            title = clean_text(row.get("title", ""))
            company = clean_text(row.get("company", ""))
            description = clean_text(row.get("description", ""))
            if not title or len(description) < 40:
                continue
            key = (title.lower(), company.lower(), description[:200].lower())
            if key in seen:
                continue
            seen.add(key)

            listed = [canonical(s) for s in re.split(r"[;,|]", row.get("skills", "") or "") if s.strip()]
            skills = sorted(set(listed) | find_skills(f"{title}. {description}"))
            jobs.append({
                "id": len(jobs),
                "title": title,
                "company": company,
                "description": description[:4000],
                "skills": skills,
                "role": (row.get("role") or "").strip(),
            })
            if max_jobs and len(jobs) >= max_jobs:
                break
    return jobs


def build_index(jobs: list[dict], embed_fn: EmbedFn, out_dir: str | Path, batch: int = 64) -> None:
    import faiss

    if not jobs:
        raise ValueError("No usable job postings were found in the CSV.")
    texts = [f"{j['title']}. {j['description'][:2000]}" for j in jobs]
    chunks = []
    for start in range(0, len(texts), batch):
        chunks.append(embed_fn(texts[start:start + batch]))
        print(f"  embedded {min(start + batch, len(texts))}/{len(texts)}", file=sys.stderr)
    vectors = np.vstack(chunks).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; fine for up to ~100k postings
    index.add(vectors)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "jobs.faiss"))
    (out / "jobs_meta.json").write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")

    # Keep the vector index self-describing so the retriever can detect stale
    # indexes after an embedding-model or schema change.
    from . import config

    manifest = {
        "format_version": 1,
        "vector_store": "FAISS IndexFlatIP",
        "similarity": "cosine",
        "vectors_l2_normalized": True,
        "embedding_model": config.EMBED_MODEL,
        "embedding_dimension": int(vectors.shape[1]),
        "vector_count": int(index.ntotal),
        "metadata_file": "jobs_meta.json",
        "index_file": "jobs.faiss",
    }
    (out / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
