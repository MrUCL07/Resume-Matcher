"""Offline tests: no network, no model download. A hashing embedder stands in for fastembed."""
import io
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import analyzer, config, extractor, main  # noqa: E402
from app.indexing import build_index, load_jobs  # noqa: E402
from app.parser import ResumeParseError, extract_text, redact_contact_info, split_sections  # noqa: E402
from app.retriever import IndexNotFoundError, Retriever  # noqa: E402
from app.scorer import rank, skill_overlap  # noqa: E402
from app.skills import canonical, find_skills  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedder (256 dims, unit length)."""
    out = np.zeros((len(texts), 256), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9+#.]+", text.lower()):
            out[row, zlib.crc32(word.encode()) % 256] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


# ---- skills -----------------------------------------------------------------
def test_skill_matching_and_aliases():
    found = find_skills("Built APIs with Python, FastAPI and Postgres; deployed on k8s. Wrote JavaScript too.")
    assert {"python", "fastapi", "postgresql", "kubernetes", "javascript"} <= found
    assert "java" not in found  # 'java' must not match inside 'javascript'
    assert canonical("K8s") == "kubernetes"


def test_ambiguous_words_need_capital_letter():
    assert "excel" not in find_skills("I like to excel at teamwork")
    assert "excel" in find_skills("Skills: Excel, SQL")
    assert "c++" in find_skills("Experienced in C++, Python")


# ---- parser -----------------------------------------------------------------
def test_redaction_keeps_year_ranges():
    text = "Mail me at a.b@example.com or +91 98765 43210. Worked 2019 - 2023. See linkedin.com/in/someone"
    out = redact_contact_info(text)
    assert "@" not in out and "98765" not in out and "linkedin" not in out
    assert "2019 - 2023" in out


def test_sections_including_inline_headings():
    text = "Jane Doe\nSUMMARY\nBackend dev.\nTechnical Skills: Python, SQL\nEXPERIENCE\nDid things at Acme.\nEducation\nB.Tech"
    sections = split_sections(text)
    assert sections["skills"] == "Python, SQL"
    assert "Acme" in sections["experience"]
    assert sections["header"] == "Jane Doe"


def test_docx_roundtrip_and_too_short_file():
    import docx

    document = docx.Document()
    document.add_paragraph("SKILLS")
    document.add_paragraph("Python, SQL, Docker, FastAPI, PostgreSQL " * 6)
    buffer = io.BytesIO()
    document.save(buffer)
    assert "Python" in extract_text("cv.docx", buffer.getvalue())

    with pytest.raises(ResumeParseError):
        extract_text("tiny.txt", b"hello")
    with pytest.raises(ResumeParseError):
        extract_text("cv.exe", b"x" * 500)


# ---- extraction: hallucinated skills must be dropped -------------------------
RESUME = "SKILLS\nPython, FastAPI, Docker\nEXPERIENCE\nBuilt REST APIs in FastAPI for a payments team. " * 3


def test_extractor_drops_unverifiable_skills(monkeypatch):
    fake = {"skills": [
        {"name": "Python", "evidence": "Python, FastAPI, Docker"},
        {"name": "Kubernetes", "evidence": "Managed 40 Kubernetes clusters"},  # not in the resume
        {"name": "FastAPI", "evidence": "a quote that is slightly wrong"},  # real skill, bad quote
    ], "years_experience": 3, "job_titles": [], "education": [], "projects": []}
    monkeypatch.setattr(extractor, "llm_available", lambda: True)
    monkeypatch.setattr(extractor, "call_llm_json", lambda prompt: fake)
    profile = extractor.extract_profile(RESUME)
    names = {s.name for s in profile.skills}
    assert "kubernetes" not in names
    assert {"python", "fastapi", "docker"} <= names
    assert profile.unverified_dropped == 1 and profile.source == "llm"


def test_extractor_falls_back_to_rules_without_llm(monkeypatch):
    monkeypatch.setattr(extractor, "llm_available", lambda: False)
    profile = extractor.extract_profile(RESUME)
    assert profile.source == "rules" and "python" in {s.name for s in profile.skills}


# ---- scoring ----------------------------------------------------------------
def test_skill_overlap_and_ranking():
    overlap, matched, missing = skill_overlap({"python", "sql"}, ["python", "sql", "docker", "aws"])
    assert overlap == 0.5 and matched == ["python", "sql"] and missing == ["aws", "docker"]

    jobs = [
        {"job": {"id": 0, "title": "A", "company": "x", "description": "d", "skills": ["java", "spring boot"]}, "embed_sim": 0.80},
        {"job": {"id": 1, "title": "B", "company": "x", "description": "d", "skills": ["python", "sql"]}, "embed_sim": 0.78},
    ]
    ranked = rank(jobs, {"python", "sql"}, top_n=2)
    assert ranked[0]["id"] == 1  # skill overlap outweighs a tiny similarity edge
    assert ranked[0]["embed_similarity"] == 0.78
    assert ranked[0]["embedding_score_pct"] == 78.0
    assert ranked[0]["cosine_distance"] == pytest.approx(0.22)
    assert ranked[0]["euclidean_distance"] == pytest.approx((0.44) ** 0.5, abs=0.0001)


# ---- analyzer: unsupported claims are removed --------------------------------
def test_analyzer_verification(monkeypatch):
    profile = extractor.ResumeProfile(skills=[extractor.SkillItem(name="python", evidence="Python")])
    match = {"title": "Backend Engineer", "company": "Acme", "matched_skills": ["python"], "missing_skills": ["aws"],
             "description": "We need Python and AWS experience for our APIs."}
    resume_text = "Built services in Python for three years."
    llm_output = {
        "fit_summary": "Good fit.",
        "matched_skills": [
            {"skill": "python", "resume_evidence": "Built services in Python", "jd_evidence": "need Python and AWS"},
            {"skill": "go", "resume_evidence": "Wrote Go microservices", "jd_evidence": ""},  # invented
        ],
        "missing_skills": [
            {"skill": "AWS", "jd_evidence": "Python and AWS experience"},
            {"skill": "Python", "jd_evidence": "need Python"},  # contradicts the resume
        ],
        "suggestions": ["a", "b", "c", "d"],
    }
    monkeypatch.setattr(analyzer, "llm_available", lambda: True)
    monkeypatch.setattr(analyzer, "call_llm_json", lambda prompt, system: llm_output)
    result = analyzer.analyze_one(profile, resume_text, match)
    assert [m["skill"] for m in result["matched_skills"]] == ["python"]
    assert [m["skill"] for m in result["missing_skills"]] == ["AWS"]
    assert result["unsupported_claims_removed"] == 2 and len(result["suggestions"]) == 3




def test_vector_index_manifest_and_model_validation(tmp_path, monkeypatch):
    jobs = [{
        "id": 0, "title": "Python Engineer", "company": "Example",
        "description": "Build Python APIs and backend services for customers.",
        "skills": ["python"], "role": "backend",
    }]
    monkeypatch.setattr(config, "EMBED_MODEL", "test-embedding-model")
    build_index(jobs, fake_embed, tmp_path)

    manifest = __import__("json").loads((tmp_path / "index_manifest.json").read_text())
    assert manifest["embedding_model"] == "test-embedding-model"
    assert manifest["vector_count"] == 1
    assert manifest["similarity"] == "cosine"
    assert manifest["vectors_l2_normalized"] is True

    retriever = Retriever(index_dir=tmp_path, embed_fn=fake_embed)
    assert retriever.search({"skills": "Python"}, "Python backend engineer", k=1)[0]["job"]["id"] == 0

    monkeypatch.setattr(config, "EMBED_MODEL", "different-model")
    with pytest.raises(IndexNotFoundError, match="Rebuild the index"):
        Retriever(index_dir=tmp_path, embed_fn=fake_embed)


# ---- end to end through the API ----------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    csv_path = ROOT / "data" / "jobs_sample.csv"
    if not csv_path.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_data.py")], check=True)
    build_index(load_jobs(csv_path), fake_embed, tmp_path)
    monkeypatch.setattr(main, "_retriever", Retriever(index_dir=tmp_path, embed_fn=fake_embed))
    monkeypatch.setattr(config, "RATE_LIMIT_PER_HOUR", 1000)
    monkeypatch.setattr(extractor, "llm_available", lambda: False)
    monkeypatch.setattr(analyzer, "llm_available", lambda: False)
    return TestClient(main.app)


def test_api_end_to_end(client):
    resume = (ROOT / "eval" / "resumes" / "devops.txt").read_bytes()
    response = client.post("/match", files={"file": ("devops.txt", resume, "text/plain")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["matches"]) == config.TOP_N_RESULTS
    assert body["matches"][0]["role"] == "devops"
    assert "analysis" in body["matches"][0]
    assert "cosine_distance" in body["matches"][0]
    assert "euclidean_distance" in body["matches"][0]
    assert "embedding_score_pct" in body["matches"][0]
    assert "example.com" not in response.text and "98123" not in response.text  # contact details never echoed


def test_api_rejects_bad_uploads(client):
    assert client.post("/match", files={"file": ("cv.exe", b"x" * 300, "application/octet-stream")}).status_code == 400
    assert client.post("/match", files={"file": ("cv.pdf", b"not a pdf" * 50, "application/pdf")}).status_code == 400
    too_big = b"a" * (config.MAX_UPLOAD_BYTES + 10)
    assert client.post("/match", files={"file": ("cv.txt", too_big, "text/plain")}).status_code == 413
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
