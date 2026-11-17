"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "index"))
STATIC_DIR = BASE_DIR / "static"
# The embedding model is downloaded here at build time (see scripts/warm_model.py)
EMBED_CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", BASE_DIR / ".fastembed_cache"))

# ---- Embeddings -----------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Cosine similarities are mapped from [SIM_FLOOR, SIM_CEIL] to [0, 1] before blending.
# These values suit bge-small; re-tune them if you switch embedding models.
SIM_FLOOR = float(os.getenv("SIM_FLOOR", "0.45"))
SIM_CEIL = float(os.getenv("SIM_CEIL", "0.85"))

# ---- LLM (hosted API, so Render only needs a small instance) ----------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" or "groq"
# Model names change over time. Check your provider's docs and override with LLM_MODEL.
DEFAULT_MODELS = {"gemini": "gemini-2.0-flash", "groq": "llama-3.3-70b-versatile"}
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(LLM_PROVIDER, "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# ---- Matching -------------------------------------------------------------
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))  # candidates pulled from FAISS
TOP_N_RESULTS = int(os.getenv("TOP_N_RESULTS", "5"))  # matches returned to the user
ANALYZE_TOP = int(os.getenv("ANALYZE_TOP", "3"))  # matches that get an LLM gap analysis
W_EMBED = float(os.getenv("W_EMBED", "0.6"))
W_SKILL = float(os.getenv("W_SKILL", "0.4"))

# ---- API limits -----------------------------------------------------------
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "2")) * 1024 * 1024)
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))  # per IP, protects your API quota
