"""Resume parsing: file bytes -> clean text -> contact-info redaction -> sections."""
import io
import re


class ResumeParseError(ValueError):
    """Raised when a resume can't be read. The message is safe to show to the user."""


MIN_CHARS = 150
MAX_PDF_PAGES = 6


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------
def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _read_pdf(data)
    elif name.endswith(".docx"):
        text = _read_docx(data)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ResumeParseError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")

    text = _clean(text)
    if len(text) < MIN_CHARS:
        raise ResumeParseError(
            "Could not read enough text from this file. If it is a scanned or image-based "
            "resume, please upload a text-based version."
        )
    return text


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ResumeParseError("This PDF is password-protected.")
        return "\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PDF_PAGES])
    except ResumeParseError:
        raise
    except Exception as exc:  # pypdf raises many different exception types
        raise ResumeParseError("This PDF could not be read.") from exc


def _read_docx(data: bytes) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ResumeParseError("This DOCX could not be read.") from exc

    parts = [p.text for p in document.paragraphs]
    for table in document.tables:  # many resume templates are built from tables
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                value = cell.text.strip()
                if value and value not in cells:  # merged cells repeat their text
                    cells.append(value)
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[•●▪■◦‣∙]", "-", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Privacy: strip contact details before anything is sent to an embedding model or LLM
# --------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github)\.com/\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)")


def redact_contact_info(text: str) -> str:
    text = EMAIL_RE.sub("[email]", text)
    text = URL_RE.sub("[link]", text)

    def _phone(match: re.Match) -> str:
        # Need 9+ digits, so year ranges like "2019 - 2023" are left alone.
        return "[phone]" if sum(c.isdigit() for c in match.group()) >= 9 else match.group()

    return PHONE_RE.sub(_phone, text)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
HEADINGS = {
    "summary": ["summary", "profile", "objective", "about me", "professional summary", "career objective"],
    "skills": ["skills", "technical skills", "key skills", "core competencies", "technologies",
               "tech stack", "skills and tools", "skills and technologies"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "internships", "internship experience"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "education": ["education", "academic background", "qualifications", "academics"],
    "certifications": ["certifications", "certificates", "courses", "licenses"],
}
HEADING_LOOKUP = {alias: key for key, aliases in HEADINGS.items() for alias in aliases}
# "Technical Skills: Python, SQL" (heading and content on one line)
INLINE_RE = re.compile(r"^([A-Za-z &/]{3,35}?)\s*[:\-–—]\s*(\S.*)$")


def _heading_key(line: str) -> str | None:
    cleaned = re.sub(r"[^a-z& ]", "", line.lower()).replace("&", "and")
    return HEADING_LOOKUP.get(re.sub(r"\s+", " ", cleaned).strip())


def split_sections(text: str) -> dict[str, str]:
    """Split a resume into sections keyed by summary/skills/experience/projects/education/...

    Text before the first heading (usually name and contact details) goes under "header".
    """
    blocks: list[tuple[str, list[str]]] = [("header", [])]
    for line in text.split("\n"):
        stripped = line.strip()
        key = _heading_key(stripped) if 0 < len(stripped) <= 40 else None
        inline = None
        if key is None:
            match = INLINE_RE.match(stripped)
            if match and _heading_key(match.group(1)):
                key, inline = _heading_key(match.group(1)), match.group(2)
        if key:
            blocks.append((key, [inline] if inline else []))
        else:
            blocks[-1][1].append(line)

    sections: dict[str, str] = {}
    for key, lines in blocks:
        body = "\n".join(lines).strip()
        if body:
            sections[key] = (sections[key] + "\n" + body) if key in sections else body
    return sections
