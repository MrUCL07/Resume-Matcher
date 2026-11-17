"""Build the FAISS index from a job postings CSV.

Usage:
    python scripts/build_index.py                              # uses data/jobs_sample.csv
    python scripts/build_index.py --csv data/raw/postings.csv --max-jobs 5000

CSV columns: title, company, description (required); skills, role (optional).
Run this on your laptop, then commit the generated index/ folder.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.embedding import get_embedder  # noqa: E402
from app.indexing import build_index, load_jobs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(config.BASE_DIR / "data" / "jobs_sample.csv"))
    parser.add_argument("--out", default=str(config.INDEX_DIR))
    parser.add_argument("--max-jobs", type=int, default=5000,
                        help="Cap the number of postings so the index stays small enough for a free Render instance")
    args = parser.parse_args()

    jobs = load_jobs(args.csv, max_jobs=args.max_jobs)
    print(f"Loaded {len(jobs)} unique job postings from {args.csv}")
    build_index(jobs, get_embedder(), args.out)
    print(f"Index written to {args.out}")


if __name__ == "__main__":
    main()
