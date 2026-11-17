"""Turn resume text into a structured profile. LLM first, keyword rules as a safety net.

Every skill the LLM claims must be backed by a quote that really appears in the
resume; otherwise it is dropped. This is the main defence against hallucinated skills.
"""
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from .llm import LLMError, call_llm_json, llm_available
from .skills import canonical, extract_skills_with_lines


log = logging.getLogger("resume-matcher")


class SkillItem(BaseModel):
    name: str
    evidence: str = ""


class ResumeProfile(BaseModel):
    skills: list[SkillItem] = Field(default_factory=list)
    years_experience: float | None = None
    job_titles: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    source: str = "llm"  # "llm" or "rules"
    unverified_dropped: int = 0
    llm_error: str = ""  # why the LLM wasn't used: "no_key" or a short, content-free error message


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """True if `quote` (ignoring case and whitespace) really appears in `text`."""
    q = norm(quote)
    return len(q) >= 3 and q in norm(text)


PROMPT = """Extract a structured profile from the resume below.

Return JSON with exactly these keys:
{{"skills": [{{"name": "<skill>", "evidence": "<exact quote>"}}],
  "years_experience": <number or null>,
  "job_titles": ["<title>"],
  "education": ["<degree, institution>"],
  "projects": ["<short project title>"]}}

Rules:
- Only list a skill if the resume explicitly mentions it. Do not infer skills.
- "evidence" must be a short quote (max 20 words) copied exactly from the resume.
- Do not include the person's name or any contact details.
- "years_experience" is total professional experience in years (count internships as half). Use null if unclear.
- Keep "projects" to at most 8 short titles.

RESUME:
\"\"\"
{resume}
\"\"\"
"""


def _rules_profile(text: str) -> ResumeProfile:
    found = extract_skills_with_lines(text)
    return ResumeProfile(
        skills=[SkillItem(name=name, evidence=line) for name, line in found.items()],
        source="rules",
    )


def extract_profile(text: str) -> ResumeProfile:
    rule_skills = extract_skills_with_lines(text)
    if not llm_available():
        log.warning("No LLM API key found; using keyword matching. Add it to .env and restart the server.")
        profile = _rules_profile(text)
        profile.llm_error = "no_key"
        return profile

    data: ResumeProfile | None = None
    last_error = ""
    for _ in range(2):  # validate, retry once
        try:
            data = ResumeProfile.model_validate(call_llm_json(PROMPT.format(resume=text[:12000])))
            break
        except LLMError as exc:
            last_error = str(exc)  # e.g. "LLM call failed (HTTP 404)"; never contains resume text or keys
        except (ValidationError, ValueError):
            last_error = "The AI returned an unexpected format"
    if data is None:
        log.warning("LLM extraction failed: %s", last_error)
        profile = _rules_profile(text)
        profile.llm_error = last_error
        return profile

    verified: dict[str, SkillItem] = {}
    dropped = 0
    for item in data.skills:
        name = canonical(item.name)
        if not name or name in verified:
            continue
        if quote_in_text(item.evidence, text):
            verified[name] = SkillItem(name=name, evidence=item.evidence.strip()[:200])
        elif name in rule_skills:  # claimed evidence was off, but the skill really is there
            verified[name] = SkillItem(name=name, evidence=rule_skills[name])
        else:
            dropped += 1

    for name, line in rule_skills.items():  # add anything the LLM missed
        verified.setdefault(name, SkillItem(name=name, evidence=line))

    years = data.years_experience
    if years is not None:
        years = max(0.0, min(float(years), 50.0))

    return ResumeProfile(
        skills=list(verified.values()),
        years_experience=years,
        job_titles=[t.strip() for t in data.job_titles if t.strip()][:6],
        education=[e.strip() for e in data.education if e.strip()][:4],
        projects=[p.strip() for p in data.projects if p.strip()][:8],
        source="llm",
        unverified_dropped=dropped,
    )
