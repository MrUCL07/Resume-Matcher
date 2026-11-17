"""Compare three ranking methods on labelled resumes: TF-IDF baseline, embeddings only, hybrid.

Each resume in eval/resumes/ is labelled in eval/labels.json with the job `role` it should match.
A job is "relevant" for a resume when its role equals the label.

Usage: python scripts/evaluate.py [--k 5]
Metrics: Recall@k, Precision@1 and MRR (mean reciprocal rank of the first relevant job).
Skills for the hybrid method come from the keyword vocabulary, so the evaluation is free and deterministic.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from app import config  # noqa: E402
from app.parser import redact_contact_info, split_sections  # noqa: E402
from app.retriever import Retriever  # noqa: E402
from app.scorer import rank  # noqa: E402
from app.skills import extract_skills_with_lines  # noqa: E402

EVAL_DIR = config.BASE_DIR / "eval"


def metrics(ranked_ids: list[int], relevant: set[int], k: int) -> dict:
    recall = len(set(ranked_ids[:k]) & relevant) / len(relevant)
    precision_at_1 = 1.0 if ranked_ids[0] in relevant else 0.0
    rr = next((1 / (i + 1) for i, job_id in enumerate(ranked_ids) if job_id in relevant), 0.0)
    return {"recall": recall, "p1": precision_at_1, "mrr": rr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    retriever = Retriever()
    jobs = retriever.jobs
    labels = json.loads((EVAL_DIR / "labels.json").read_text(encoding="utf-8"))
    n_jobs = len(jobs)

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    job_matrix = tfidf.fit_transform([f"{j['title']}. {j['description']}" for j in jobs])

    scores: dict[str, list[dict]] = {"TF-IDF baseline": [], "Embeddings only": [], "Hybrid (embeddings + skills)": []}
    for filename, role in labels.items():
        relevant = {j["id"] for j in jobs if j["role"] == role}
        if not relevant:
            print(f"Skipping {filename}: no jobs with role '{role}'")
            continue
        text = redact_contact_info((EVAL_DIR / "resumes" / filename).read_text(encoding="utf-8"))
        sections = split_sections(text)

        sims = (job_matrix @ tfidf.transform([text]).T).toarray().ravel()
        tfidf_ranked = [int(i) for i in np.argsort(-sims)]

        candidates = retriever.search(sections, text, k=n_jobs)
        embed_ranked = [c["job"]["id"] for c in candidates]
        hybrid_ranked = [r["id"] for r in rank(candidates, set(extract_skills_with_lines(text)), top_n=n_jobs)]

        for name, ranked in zip(scores, (tfidf_ranked, embed_ranked, hybrid_ranked)):
            scores[name].append(metrics(ranked, relevant, args.k))

    n_resumes = len(next(iter(scores.values())))
    lines = [f"Evaluated {n_resumes} resumes against {n_jobs} job postings (k={args.k}).", "",
             f"| Method | Recall@{args.k} | Precision@1 | MRR |", "|---|---|---|---|"]
    for name, rows in scores.items():
        avg = {key: float(np.mean([r[key] for r in rows])) for key in ("recall", "p1", "mrr")}
        lines.append(f"| {name} | {avg['recall']:.2f} | {avg['p1']:.2f} | {avg['mrr']:.2f} |")
    report = "\n".join(lines)
    print(report)
    (EVAL_DIR / "results.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
