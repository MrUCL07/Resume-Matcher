"""Hybrid score = W_EMBED * embedding similarity + W_SKILL * skill overlap.

Embeddings capture overall topical fit; skill overlap catches the hard requirements
that embeddings blur together (e.g. "Python" vs "Java" can embed very similarly).
"""
from . import config


def skill_overlap(resume_skills: set[str], job_skills: list[str]) -> tuple[float, list[str], list[str]]:
    """Fraction of the job's skills that the resume has, plus the matched and missing lists."""
    if not job_skills:
        return 0.0, [], []
    matched = sorted(s for s in job_skills if s in resume_skills)
    missing = sorted(s for s in job_skills if s not in resume_skills)
    return len(matched) / len(job_skills), matched, missing


def scale_similarity(sim: float) -> float:
    """Map a raw cosine similarity onto 0..1 using a fixed window (see config)."""
    span = config.SIM_CEIL - config.SIM_FLOOR
    return max(0.0, min(1.0, (sim - config.SIM_FLOOR) / span))


def hybrid_score(embed_sim: float, overlap: float,
                 w_embed: float = config.W_EMBED, w_skill: float = config.W_SKILL) -> float:
    return w_embed * scale_similarity(embed_sim) + w_skill * overlap


def rank(candidates: list[dict], resume_skills: set[str], top_n: int,
         w_embed: float = config.W_EMBED, w_skill: float = config.W_SKILL) -> list[dict]:
    results = []
    for cand in candidates:
        job = cand["job"]
        overlap, matched, missing = skill_overlap(resume_skills, job.get("skills", []))
        # FAISS stores L2-normalized vectors, so inner product is cosine similarity.
        # For unit vectors: cosine distance = 1 - cosine similarity and
        # Euclidean distance = sqrt(2 - 2 * cosine similarity).
        cosine_similarity = max(-1.0, min(1.0, float(cand["embed_sim"])))
        cosine_distance = 1.0 - cosine_similarity
        euclidean_distance = (2.0 * cosine_distance) ** 0.5
        results.append({
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "role": job.get("role", ""),
            "score": round(100 * hybrid_score(cand["embed_sim"], overlap, w_embed, w_skill), 1),
            "embed_similarity": round(cosine_similarity, 4),
            "embedding_score_pct": round(max(0.0, min(1.0, cosine_similarity)) * 100, 2),
            "cosine_distance": round(cosine_distance, 4),
            "euclidean_distance": round(euclidean_distance, 4),
            "skill_overlap": round(overlap, 3),
            "matched_skills": matched,
            "missing_skills": missing,
            "description": job["description"],
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]
