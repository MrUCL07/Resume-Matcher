"""A small skills vocabulary with alias handling.

Used in three places: tagging job descriptions at index time, a keyword-based
fallback when the LLM is unavailable, and double-checking the LLM's claims.
Extend SKILLS freely; keys are the canonical names shown in the UI.
"""
import re

SKILLS: dict[str, list[str]] = {
    # languages
    "python": [], "java": [], "javascript": ["js", "es6"], "typescript": ["ts"],
    "c++": ["cpp"], "c#": ["csharp"], "golang": [], "rust": [], "php": [], "ruby": [],
    "kotlin": [], "swift": [], "scala": [], "bash": ["shell scripting"],
    # data stores
    "sql": [], "postgresql": ["postgres"], "mysql": [], "mongodb": ["mongo"], "redis": [],
    "elasticsearch": [], "snowflake": [], "bigquery": [],
    # web
    "html": ["html5"], "css": ["css3"], "react": ["reactjs", "react.js"], "angular": [],
    "vue": ["vue.js", "vuejs"], "node.js": ["nodejs"], "express.js": ["expressjs"],
    "next.js": ["nextjs"], "tailwind css": ["tailwind"], "django": [], "flask": [],
    "fastapi": [], "spring boot": [], ".net": ["dotnet"], "rest api": ["rest apis", "restful api", "restful apis", "restful"],
    "graphql": [], "microservices": [], "grpc": [],
    # cloud / devops
    "aws": ["amazon web services"], "azure": [], "gcp": ["google cloud"], "docker": [],
    "kubernetes": ["k8s"], "terraform": [], "ansible": [], "jenkins": [],
    "github actions": [], "ci/cd": ["cicd"], "linux": [], "git": [], "nginx": [],
    "prometheus": [], "grafana": [],
    # data / ML
    "pandas": [], "numpy": [], "scikit-learn": ["sklearn", "scikit learn"], "tensorflow": [],
    "pytorch": [], "keras": [], "machine learning": ["ml"], "deep learning": [],
    "nlp": ["natural language processing"], "computer vision": [],
    "llm": ["llms", "large language models"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "langchain": [], "llamaindex": [], "hugging face": ["huggingface", "transformers"],
    "statistics": ["statistical analysis"], "data analysis": ["data analytics"],
    "data visualization": ["data visualisation"], "a/b testing": ["ab testing"],
    "feature engineering": [], "mlops": [],
    # analytics / data engineering
    "tableau": [], "power bi": ["powerbi"], "excel": ["microsoft excel", "ms excel", "advanced excel"],
    "looker": [], "spark": ["pyspark", "apache spark"], "airflow": [], "kafka": [],
    "etl": [], "dbt": [], "hadoop": [],
    # practices / tools
    "agile": ["scrum"], "unit testing": ["pytest", "jest"], "figma": [],
}

# Ordinary English words: only match these when written with a capital letter.
CASE_SENSITIVE = {"excel", "swift", "spark"}

ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SKILLS.items():
    ALIAS_TO_CANONICAL[_canonical] = _canonical
    for _alias in _aliases:
        ALIAS_TO_CANONICAL[_alias] = _canonical


def canonical(name: str) -> str:
    """Map any alias ('k8s', 'Postgres') to its canonical form; unknown names are just lowercased."""
    key = re.sub(r"\s+", " ", name.strip().lower())
    return ALIAS_TO_CANONICAL.get(key, key)


def _build_pattern(skill: str, aliases: list[str]) -> re.Pattern:
    def alt(terms):
        return "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))

    if skill in CASE_SENSITIVE:
        body = f"(?i:{alt(aliases)})|{re.escape(skill.capitalize())}" if aliases else re.escape(skill.capitalize())
    else:
        body = f"(?i:{alt([skill] + aliases)})"
    # Not preceded/followed by a word character, '+' or '#', so 'java' does not match 'javascript'.
    return re.compile(rf"(?<![A-Za-z0-9+#.])(?:{body})(?![A-Za-z0-9+#])")


PATTERNS = {skill: _build_pattern(skill, aliases) for skill, aliases in SKILLS.items()}


def find_skills(text: str) -> set[str]:
    """Canonical skill names mentioned anywhere in the text."""
    return {skill for skill, pattern in PATTERNS.items() if pattern.search(text)}


def extract_skills_with_lines(text: str) -> dict[str, str]:
    """Canonical skill -> the first line of the text that mentions it (used as evidence)."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    found: dict[str, str] = {}
    for skill, pattern in PATTERNS.items():
        for line in lines:
            if pattern.search(line):
                found[skill] = line[:200]
                break
    return found
