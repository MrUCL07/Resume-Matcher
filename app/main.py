"""FastAPI app: POST /match takes a resume file and returns matching jobs with gap analysis."""
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .analyzer import analyze_matches
from .upload_check import check_magic_bytes
from .extractor import extract_profile
from .llm import llm_available
from .parser import ResumeParseError, extract_text, redact_contact_info, split_sections
from .retriever import IndexNotFoundError, Retriever
from .scorer import rank
from .skills import canonical

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("resume-matcher")
# NOTE: resume text is never logged anywhere in this app.

_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_retriever()  # load the model and index at startup, not on the first request
        log.info("Index and embedding model loaded.")
    except IndexNotFoundError as exc:
        log.warning("%s", exc)
    yield


app = FastAPI(title="Resume to Job Matcher", lifespan=lifespan)

# ---- very small in-memory rate limiter (protects your free LLM quota) --------
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")  # Render sits behind a proxy
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    with _hits_lock:
        window = _hits[ip]
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= config.RATE_LIMIT_PER_HOUR:
            raise HTTPException(429, "Too many requests. Please try again later.")
        window.append(now)


# ---- routes -----------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": _retriever is not None,
        "llm_configured": llm_available(),  # True/False only; the key itself is never exposed
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
    }


@app.post("/match")
def match(request: Request, file: UploadFile = File(...)):
    _check_rate_limit(_client_ip(request))

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        raise HTTPException(400, "Please upload a PDF, DOCX or TXT file.")
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if not check_magic_bytes(suffix, data):
        raise HTTPException(400, "The file content does not match its extension.")

    try:
        raw_text = extract_text(filename, data)
    except ResumeParseError as exc:
        raise HTTPException(422, str(exc))

    text = redact_contact_info(raw_text)  # from here on, only the redacted text is used
    sections = split_sections(text)
    profile = extract_profile(text)

    try:
        retriever = get_retriever()
    except IndexNotFoundError:
        raise HTTPException(503, "The job index has not been built yet.")

    candidates = retriever.search(sections, text, k=config.TOP_K_RETRIEVE)
    resume_skills = {canonical(s.name) for s in profile.skills}
    top = rank(candidates, resume_skills, config.TOP_N_RESULTS)
    analyze_matches(profile, text, top, config.ANALYZE_TOP)

    warnings = []
    if profile.llm_error == "no_key":
        warnings.append("No AI API key was found, so only keyword matching was used. Add GEMINI_API_KEY (or "
                        "GROQ_API_KEY) to your .env file and restart the server.")
    elif profile.source == "rules":
        warnings.append(f"The AI model call failed ({profile.llm_error or 'unknown error'}), so only keyword "
                        "matching was used. Check your API key and the LLM_MODEL setting.")
    if profile.unverified_dropped:
        warnings.append(f"{profile.unverified_dropped} skill(s) the AI suggested were removed because they could not be verified in your resume.")

    for item in top:
        item["description"] = item["description"][:400]  # preview only
    return {
        "profile": {
            "skills": [s.model_dump() for s in profile.skills],
            "years_experience": profile.years_experience,
            "job_titles": profile.job_titles,
            "education": profile.education,
            "projects": profile.projects,
        },
        "matches": top,
        "warnings": warnings,
    }


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(config.STATIC_DIR / "index.html")
