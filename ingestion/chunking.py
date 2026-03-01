from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Section:
    path: str
    title: str
    text: str


_SEC_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+(.+?)\s*$")
_SEC_WORD = re.compile(r"^\s*Section\s+(\d+(?:\.\d+)*)[:\.-]?\s*(.*?)\s*$", re.IGNORECASE)
_ARTICLE = re.compile(r"^\s*Article\s+([IVXLC]+|\d+)[:\.-]?\s*(.*?)\s*$", re.IGNORECASE)
_ALL_CAPS = re.compile(r"^\s*[A-Z][A-Z\s]{4,}\s*$")

_LIST_LINE = re.compile(
    r"^\s*(?:[-*•]|\d+\)|\([a-zA-Z]\)|\([ivxIVX]+\))\s+"
)


def chunk_all(docs: Sequence[Tuple[str, str]], *, child_max_chars: int, overlap_chars: int, min_chars: int) -> List[Dict]:
    chunks: List[Dict] = []
    for raw, doc_id in docs:
        chunks.extend(
            chunk_one(
                raw,
                doc_id,
                child_max_chars=child_max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )
        )
    return chunks


def chunk_one(raw_text: str, doc_id: str, *, child_max_chars: int, overlap_chars: int, min_chars: int) -> List[Dict]:
    title = _doc_title(raw_text, fallback=doc_id)
    sections = _split_sections(raw_text)
    if not sections:
        sections = [Section(path="root", title=title, text=raw_text.strip())]

    out: List[Dict] = []
    for sec in sections:
        parent_text = _format_parent(title, sec)
        child_texts = _make_children(sec.text, max_chars=child_max_chars, overlap_chars=overlap_chars)
        for i, child in enumerate(child_texts):
            if len(child) < min_chars:
                continue
            out.append(
                {
                    "chunk_id": f"{doc_id}:{sec.path}:c{i:03d}",
                    "doc_id": doc_id,
                    "doc_title": title,
                    "section_path": sec.path,
                    "section_title": sec.title,
                    "chunk_kind": "child",
                    "text": child.strip(),
                    "parent_text": parent_text,
                }
            )
    return out


def _doc_title(raw: str, fallback: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return fallback


def _split_sections(raw: str) -> List[Section]:
    lines = raw.splitlines()
    marks: List[Tuple[int, str, str]] = []

    for i, line in enumerate(lines):
        m = _SEC_NUM.match(line)
        if m:
            marks.append((i, m.group(1), m.group(2)))
            continue
        m = _SEC_WORD.match(line)
        if m:
            marks.append((i, m.group(1), (m.group(2) or "").strip()))
            continue
        m = _ARTICLE.match(line)
        if m:
            marks.append((i, str(m.group(1)), (m.group(2) or "").strip()))
            continue

    if not marks:
        return []

    sections: List[Section] = []
    for idx, (start, path, title) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body_lines = lines[start + 1 : end]
        body = "\n".join(body_lines).strip()
        sections.append(Section(path=path, title=title, text=body))
    return sections


def _format_parent(doc_title: str, sec: Section) -> str:
    head = f"{doc_title}\nSection {sec.path}"
    if sec.title:
        head += f": {sec.title}"
    return (head + "\n\n" + sec.text.strip()).strip()


def _make_children(text: str, *, max_chars: int, overlap_chars: int) -> List[str]:
    blocks = _split_blocks(text)
    units: List[str] = []
    for b in blocks:
        if _looks_like_list_block(b):
            units.extend(_split_list_block_if_needed(b, max_chars=max_chars))
        else:
            units.extend(_split_sentences(b))

    return _pack_units(units, max_chars=max_chars, overlap_chars=overlap_chars)


def _split_blocks(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _looks_like_list_block(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    hits = sum(1 for ln in lines if _LIST_LINE.match(ln))
    return hits >= 2


def _split_list_block_if_needed(block: str, *, max_chars: int) -> List[str]:
    if len(block) <= max_chars:
        return [block.strip()]

    items = _split_list_items(block)
    if len(items) <= 1:
        return _split_sentences(block)

    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for it in items:
        it_len = len(it) + 1
        if cur and cur_len + it_len > max_chars:
            out.append("\n".join(cur).strip())
            cur = [it]
            cur_len = it_len
        else:
            cur.append(it)
            cur_len += it_len
    if cur:
        out.append("\n".join(cur).strip())
    return out


def _split_list_items(block: str) -> List[str]:
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    items: List[str] = []
    cur: List[str] = []
    for ln in lines:
        if _LIST_LINE.match(ln) and cur:
            items.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        items.append("\n".join(cur).strip())
    return items


def _split_sentences(text: str) -> List[str]:
    s = " ".join(ln.strip() for ln in text.splitlines() if ln.strip()).strip()
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [s]


def _pack_units(units: Sequence[str], *, max_chars: int, overlap_chars: int) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for u in units:
        u = u.strip()
        if not u:
            continue
        u_len = len(u) + 1
        if cur and cur_len + u_len > max_chars:
            out.append(_join_units(cur))
            cur = _overlap(cur, overlap_chars=overlap_chars)
            cur_len = sum(len(x) + 1 for x in cur)

        cur.append(u)
        cur_len += u_len

    if cur:
        out.append(_join_units(cur))
    return [c for c in out if c.strip()]


def _join_units(units: Sequence[str]) -> str:
    if any("\n" in u for u in units):
        return "\n\n".join(units).strip()
    return " ".join(units).strip()


def _overlap(prev_units: Sequence[str], *, overlap_chars: int) -> List[str]:
    if not prev_units or overlap_chars <= 0:
        return []
    last = prev_units[-1]
    return [last] if len(last) <= overlap_chars else []

