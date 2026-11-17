"""Embed uploaded resumes in memory and search the persistent FAISS vector store."""
import json
from pathlib import Path

from . import config
from .embedding import EmbedFn


class IndexNotFoundError(RuntimeError):
    pass


QUERY_SECTIONS = ("summary", "skills", "experience", "projects")


def _split_long(text: str, limit: int = 1200, max_chunks: int = 3) -> list[str]:
    chunks, current = [], ""
    for line in text.split("\\n"):
        if current and len(current) + len(line) + 1 > limit:
            chunks.append(current.strip())
            current = ""
        current += line + "\\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks[:max_chunks]


def query_chunks(sections: dict[str, str], full_text: str) -> list[str]:
    """Chunk by resume section so each query vector represents one topic."""
    chunks: list[str] = []
    for name in QUERY_SECTIONS:
        if sections.get(name):
            chunks += _split_long(sections[name])
    return chunks or _split_long(full_text, max_chunks=4)


class Retriever:
    """Persistent vector retrieval over job postings.

    FAISS stores normalized job vectors; job metadata is stored separately in
    jobs_meta.json. Resume vectors are created at request time and never saved.
    """

    def __init__(self, index_dir: str | Path | None = None, embed_fn: EmbedFn | None = None):
        import faiss

        index_dir = Path(index_dir or config.INDEX_DIR)
        index_path = index_dir / "jobs.faiss"
        meta_path = index_dir / "jobs_meta.json"
        manifest_path = index_dir / "index_manifest.json"

        if not index_path.exists() or not meta_path.exists():
            raise IndexNotFoundError("Job vector index not found. Run: python scripts/build_index.py")

        self.index = faiss.read_index(str(index_path))
        self.jobs: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))

        if self.index.ntotal != len(self.jobs):
            raise IndexNotFoundError("FAISS index and job metadata are out of sync. Rebuild the index.")

        # Older indexes without a manifest remain readable. Newly built indexes
        # include it so incompatible embedding models are caught immediately.
        self.manifest = None
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_model = self.manifest.get("embedding_model")
            if expected_model and expected_model != config.EMBED_MODEL:
                raise IndexNotFoundError(
                    f"Index uses embedding model '{expected_model}', but configuration requests "
                    f"'{config.EMBED_MODEL}'. Rebuild it with: python scripts/build_index.py"
                )
            if self.manifest.get("vector_count") != self.index.ntotal:
                raise IndexNotFoundError("Index manifest does not match the FAISS vector count. Rebuild the index.")
            if self.manifest.get("embedding_dimension") != self.index.d:
                raise IndexNotFoundError("Index manifest dimension does not match FAISS. Rebuild the index.")

        if embed_fn is None:
            from .embedding import get_embedder
            embed_fn = get_embedder()
        self.embed = embed_fn

    def search(self, sections: dict[str, str], full_text: str, k: int = 20) -> list[dict]:
        """Return top jobs with cosine similarity, retaining each job's best resume-chunk score."""
        if self.index.ntotal == 0 or k <= 0:
            return []

        chunks = query_chunks(sections, full_text)
        if not chunks:
            return []

        query_vectors = self.embed(chunks)
        if query_vectors.ndim != 2 or query_vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension {query_vectors.shape[-1]} does not match "
                f"FAISS index dimension {self.index.d}."
            )

        k = min(k, self.index.ntotal)
        sims, ids = self.index.search(query_vectors, k)
        best: dict[int, float] = {}
        for row_sims, row_ids in zip(sims, ids):
            for sim, job_id in zip(row_sims, row_ids):
                if job_id >= 0:
                    best[int(job_id)] = max(best.get(int(job_id), -1.0), float(sim))

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [{"job": self.jobs[job_id], "embed_sim": sim} for job_id, sim in ranked]
