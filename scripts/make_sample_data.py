"""Generate a small synthetic job-postings CSV so the project runs before you download real data.

Replace data/jobs_sample.csv with a real dataset (e.g. a Kaggle job postings CSV) for a serious project.
Usage: python scripts/make_sample_data.py
"""
import csv
from pathlib import Path

FAMILIES = {
    "backend": {
        "titles": ["Backend Engineer", "Python Developer", "Software Engineer - Backend", "API Developer"],
        "resp": "design and maintain REST APIs, optimise database queries and ship reliable services",
        "skill_sets": [
            ["python", "fastapi", "postgresql", "docker", "rest api", "git"],
            ["python", "django", "mysql", "redis", "rest api", "linux"],
            ["java", "spring boot", "microservices", "mysql", "docker", "kubernetes"],
            ["node.js", "mongodb", "rest api", "aws", "docker", "ci/cd"],
        ],
    },
    "frontend": {
        "titles": ["Frontend Developer", "React Developer", "UI Engineer", "Web Developer"],
        "resp": "build responsive user interfaces, work with designers and keep the web app fast and accessible",
        "skill_sets": [
            ["javascript", "react", "html", "css", "git", "rest api"],
            ["typescript", "react", "next.js", "tailwind css", "figma", "unit testing"],
            ["javascript", "vue", "html", "css", "rest api", "agile"],
            ["typescript", "angular", "css", "unit testing", "ci/cd", "git"],
        ],
    },
    "data_scientist": {
        "titles": ["Data Scientist", "Machine Learning Engineer", "NLP Engineer", "Applied Scientist"],
        "resp": "build and evaluate machine learning models, run experiments and deploy models to production",
        "skill_sets": [
            ["python", "pandas", "scikit-learn", "sql", "statistics", "machine learning"],
            ["python", "pytorch", "deep learning", "nlp", "hugging face", "docker"],
            ["python", "llm", "rag", "langchain", "fastapi", "nlp"],
            ["python", "tensorflow", "computer vision", "mlops", "aws", "git"],
        ],
    },
    "data_analyst": {
        "titles": ["Data Analyst", "Business Analyst", "BI Developer", "Analytics Associate"],
        "resp": "turn raw data into dashboards and insights that guide business decisions",
        "skill_sets": [
            ["sql", "excel", "tableau", "data analysis", "statistics", "python"],
            ["sql", "power bi", "excel", "data visualization", "a/b testing", "pandas"],
            ["sql", "tableau", "looker", "etl", "data analysis", "excel"],
            ["python", "pandas", "sql", "data visualization", "statistics", "excel"],
        ],
    },
    "devops": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Platform Engineer"],
        "resp": "automate deployments, manage cloud infrastructure and keep services observable and reliable",
        "skill_sets": [
            ["aws", "terraform", "docker", "kubernetes", "ci/cd", "linux"],
            ["azure", "kubernetes", "ansible", "jenkins", "linux", "bash"],
            ["gcp", "terraform", "kubernetes", "prometheus", "grafana", "python"],
            ["aws", "docker", "github actions", "linux", "bash", "nginx"],
        ],
    },
}
COMPANIES = ["Northwind Labs", "Bluepeak Systems", "Kestrel Tech", "Orbit Works", "Lumen Analytics",
             "Cobalt Digital", "Fernhill Software", "Quanta Retail", "Mosaic Health", "Riverbend Finance",
             "Skyline Logistics", "Pinecone Studio", "Helix Robotics", "Granite Cloud", "Zephyr Media",
             "Atlas Mobility", "Verity Insurance", "Nimbus Games", "Harbor Energy", "Tandem Education"]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "jobs_sample.csv"
    out.parent.mkdir(exist_ok=True)
    rows, n = [], 0
    for role, family in FAMILIES.items():
        for title, skills in zip(family["titles"], family["skill_sets"]):
            company = COMPANIES[n % len(COMPANIES)]
            n += 1
            description = (f"{company} is looking for a {title} to {family['resp']}. "
                           f"Requirements: hands-on experience with {', '.join(skills)}. "
                           f"Strong problem solving and communication skills.")
            rows.append({"title": title, "company": company, "description": description,
                         "skills": ";".join(skills), "role": role})
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "company", "description", "skills", "role"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} postings to {out}")


if __name__ == "__main__":
    main()
