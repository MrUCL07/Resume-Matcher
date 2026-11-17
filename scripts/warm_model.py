"""Download the embedding model during the Render build, so the app doesn't download it at startup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedding import get_embedder  # noqa: E402

if __name__ == "__main__":
    embed = get_embedder()
    print("Model ready. Embedding dimension:", embed(["warm up"]).shape[1])
