"""Cheap upload sanity check: does the file's first bytes match its extension?"""


def check_magic_bytes(suffix: str, data: bytes) -> bool:
    if suffix == ".pdf":
        return data.lstrip()[:5] == b"%PDF-"
    if suffix == ".docx":
        return data[:2] == b"PK"  # .docx is a zip archive
    if suffix == ".txt":
        return b"\x00" not in data[:1024]  # reject binary files renamed to .txt
    return False
