"""The generation step of the RAG pipeline: a gap analysis grounded in the retrieved job text.

The LLM must quote the resume and the job description. Quotes that don't appear
in the source text are discarded, so unsupported claims never reach the user.
"""
from concurrent.futures import ThreadPoolExecutor

from .extractor import ResumeProfile, quote_in_text
from .llm import LLMError, call_llm_json, llm_available
from .skills import canonical

SYSTEM = "You are a careful career coach. Use only the text you are given. Reply with valid JSON only."

PROMPT = """Compare this resume with this job posting.

Rules:
- Use ONLY the RESUME and JOB texts below. Never invent skills or experience.
- Every "resume_evidence" must be an exact quote (max 20 words) copied from the RESUME.
- Every "jd_evidence" must be an exact quote (max 20 words) copied from the JOB.
- A skill is "missing" only if the RESUME does not mention it. If unsure, leave it out.
- Give exactly 3 specific suggestions for improving this resume for this job. Only suggest adding
  a skill if the candidate plausibly has real experience with it.

Return JSON:
{{"fit_summary": "<max 2 sentences>",
  "matched_skills": [{{"skill": "...", "resume_evidence": "...", "jd_evidence": "..."}}],
  "missing_skills": [{{"skill": "...", "jd_evidence": "..."}}],
  "suggestions": ["...", "...", "..."]}}

JOB: {title} at {company}
\"\"\"
{job_text}
\"\"\"

RESUME:
\"\"\"
{resume_text}
\"\"\"
"""


def _as_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _verify(data: dict, resume_text: str, job_text: str, resume_skills: set[str]) -> tuple[dict, int]:
    dropped = 0
    matched = []
    for item in (data.get("matched_skills") or [])[:12]:
        item = item if isinstance(item, dict) else {}
        skill, resume_quote, jd_quote = _as_text(item.get("skill")), _as_text(item.get("resume_evidence")), _as_text(item.get("jd_evidence"))
        if skill and quote_in_text(resume_quote, resume_text):
            matched.append({
                "skill": skill,
                "resume_evidence": resume_quote,
                "jd_evidence": jd_quote if quote_in_text(jd_quote, job_text) else "",
            })
        else:
            dropped += 1

    missing = []
    for item in (data.get("missing_skills") or [])[:12]:
        item = item if isinstance(item, dict) else {}
        skill, jd_quote = _as_text(item.get("skill")), _as_text(item.get("jd_evidence"))
        if not skill or not quote_in_text(jd_quote, job_text) or canonical(skill) in resume_skills:
            dropped += 1  # unsupported, or contradicts the resume
            continue
        missing.append({"skill": skill, "jd_evidence": jd_quote})

    suggestions = [_as_text(s) for s in (data.get("suggestions") or []) if _as_text(s)][:3]
    summary = _as_text(data.get("fit_summary"))[:400]
    return {"fit_summary": summary, "matched_skills": matched,
            "missing_skills": missing, "suggestions": suggestions}, dropped


def _rule_analysis(profile: ResumeProfile, match: dict) -> dict:
    """Deterministic fallback used when the LLM is unavailable or its output fails verification."""
    evidence = {canonical(s.name): s.evidence for s in profile.skills}
    matched = [{"skill": s, "resume_evidence": evidence.get(s, ""), "jd_evidence": ""} for s in match["matched_skills"]]
    missing = [{"skill": s, "jd_evidence": ""} for s in match["missing_skills"]]
    total = len(matched) + len(missing)
    summary = (f"Your resume covers {len(matched)} of the {total} skills detected in this posting."
               if total else "No specific skills could be detected in this posting.")
    suggestions = [f"If you have real experience with {m['skill']}, add a bullet showing where you used it."
                   for m in missing[:3]]
    if not suggestions:
        suggestions = ["Add measurable results (numbers, percentages, scale) to your most relevant bullets."]
    return {"fit_summary": summary, "matched_skills": matched, "missing_skills": missing,
            "suggestions": suggestions, "source": "rules", "unsupported_claims_removed": 0}


def analyze_one(profile: ResumeProfile, resume_text: str, match: dict) -> dict:
    if not llm_available():
        return _rule_analysis(profile, match)
    job_text = match["description"][:3000]
    prompt = PROMPT.format(title=match["title"], company=match["company"],
                           job_text=job_text, resume_text=resume_text[:6000])
    try:
        data = call_llm_json(prompt, SYSTEM)
    except LLMError:
        return _rule_analysis(profile, match)

    resume_skills = {canonical(s.name) for s in profile.skills}
    result, dropped = _verify(data, resume_text, job_text, resume_skills)
    if not result["matched_skills"] and not result["missing_skills"]:
        return _rule_analysis(profile, match)  # nothing survived verification
    result.update({"source": "llm", "unsupported_claims_removed": dropped})
    return result


def analyze_matches(profile: ResumeProfile, resume_text: str, matches: list[dict], analyze_top: int) -> None:
    """Attach an `analysis` dict to every match; only the top few use the LLM (saves quota and time)."""
    llm_part, rest = matches[:analyze_top], matches[analyze_top:]
    if llm_part:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for match, analysis in zip(llm_part, pool.map(lambda m: analyze_one(profile, resume_text, m), llm_part)):
                match["analysis"] = analysis
    for match in rest:
        match["analysis"] = _rule_analysis(profile, match)
