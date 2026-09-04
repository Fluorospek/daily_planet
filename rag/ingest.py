from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from bs4 import BeautifulSoup

@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)

def _parse_frontmatter(raw):

    if not raw.startswith("---"):
        return {}, raw.strip()

    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw.strip()

    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

    return meta, parts[2].strip()


def load_markdown(path):

    raw = Path(path).read_text(encoding="utf-8")
    meta, text = _parse_frontmatter(raw)
    meta.setdefault("source", Path(path).name)
    return Document(content=text, metadata=meta)

def load_html(path):

    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")
    text = soup.get_text(separator="\n")
    clean = "\n".join(line.strip() for line in text.splitlines() if line .strip())
    return Document(text = clean, metadata={"source": Path(path).name})