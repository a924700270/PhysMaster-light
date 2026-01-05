import os
import re
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path

_DISK_KB_INDEX = None  # lazy-built: list[dict]

def _kb_roots() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    kb_root = repo_root() / "knowledge_base"
    return {
        "local": kb_root / "local_library",
        "global": kb_root / "global_library",
        "prior": kb_root / "global_prior",
        "methodology": kb_root / "global_methodology"
    }


def _tokenize_query(q: str) -> list[str]:
    q = (q or "").strip().lower()
    if not q:
        return []
    # English tokens + Chinese blocks
    toks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", q)
    # drop ultra-short latin tokens to reduce noise
    cleaned = []
    for t in toks:
        if re.fullmatch(r"[a-z0-9_]+", t) and len(t) <= 2:
            continue
        cleaned.append(t)
    return cleaned


def _extract_search_text(obj, *, max_chars: int = 20000) -> str:
    """
    Best-effort extraction:
    - prioritize title/abstract/core_knowledge(qualitative/quantitative)
    - recursively collect strings, but skip obviously huge id lists
    """
    parts: list[str] = []

    def add(s: str):
        if not s:
            return
        s = str(s).strip()
        if not s:
            return
        parts.append(s)

    def walk(x, depth: int = 0):
        if len(" ".join(parts)) > max_chars:
            return
        if depth > 8:
            return
        if isinstance(x, str):
            add(x)
            return
        if isinstance(x, (int, float, bool)) or x is None:
            return
        if isinstance(x, list):
            # skip huge numeric/id lists
            if len(x) > 500 and all(isinstance(v, (int, float, str)) for v in x):
                return
            for v in x[:200]:
                walk(v, depth + 1)
            return
        if isinstance(x, dict):
            # priority keys first
            for k in ["title", "abstract", "core_knowledge", "qualitative", "quantitative", "summary", "takeaways"]:
                if k in x:
                    walk(x.get(k), depth + 1)
            # then everything else except obvious junk
            for k, v in x.items():
                if k in ("touch_ids", "ids", "references", "bib", "citations_list"):
                    continue
                if k in ("title", "abstract", "core_knowledge", "qualitative", "quantitative", "summary", "takeaways"):
                    continue
                walk(v, depth + 1)

    walk(obj, 0)
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _iter_files(root: Path, exts: tuple[str, ...]) -> Iterable[Path]:
    if not root or not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _build_disk_index(force: bool = False) -> list[dict]:
    global _DISK_KB_INDEX
    if _DISK_KB_INDEX is not None and not force:
        return _DISK_KB_INDEX

    roots = _kb_roots()
    index: list[dict] = []

    # 1) local/global/prior: JSON
    for source in ("local", "global", "prior"):
        root = roots.get(source)
        if not root or not root.exists():
            continue
        for fp in _iter_files(root, (".json",)):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # If it contains extra.recall_papers/top_papers, index each candidate for finer granularity
            candidates = []
            if isinstance(data, dict):
                extra = data.get("extra") or {}
                if isinstance(extra, dict):
                    rp = extra.get("recall_papers") or extra.get("top_papers")
                    if isinstance(rp, list):
                        candidates = [c for c in rp if isinstance(c, dict)]

            if candidates:
                for i, c in enumerate(candidates[:200]):
                    title = c.get("title", "") or data.get("title", "")
                    text = _extract_search_text(c)
                    if not text and title:
                        text = title
                    index.append(
                        {
                            "source": source,
                            "path": f"{fp.as_posix()}#cand{i}",
                            "title": title,
                            "text": text,
                            "text_lower": (text or "").lower(),
                        }
                    )
            else:
                title = data.get("title", "") if isinstance(data, dict) else ""
                text = _extract_search_text(data)
                if not text and title:
                    text = title
                index.append(
                    {
                        "source": source,
                        "path": fp.as_posix(),
                        "title": title,
                        "text": text,
                        "text_lower": (text or "").lower(),
                    }
                )

    # 2) methodology: manual notes (.md/.txt), fallback legacy global_case
    meth_root = roots.get("methodology")
    legacy_root = roots.get("case_legacy")
    chosen = meth_root if (meth_root and meth_root.exists()) else legacy_root

    if chosen and chosen.exists():
        for fp in _iter_files(chosen, (".md", ".txt", ".json")):
            try:
                if fp.suffix.lower() == ".json":
                    with fp.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    title = (data.get("title") if isinstance(data, dict) else "") or fp.stem
                    text = _extract_search_text(data)
                else:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    title = fp.stem
                text = (text or "")[:20000]
            except Exception:
                continue

            index.append(
                {
                    "source": "methodology",
                    "path": fp.as_posix(),
                    "title": title,
                    "text": text,
                    "text_lower": (text or "").lower(),
                }
            )

    _DISK_KB_INDEX = index
    return _DISK_KB_INDEX


def _make_excerpt(text: str, q_lower: str, tokens: list[str], *, window: int = 220) -> str:
    if not text:
        return ""
    t_lower = text.lower()
    pos = -1
    if q_lower:
        pos = t_lower.find(q_lower)
    if pos < 0:
        for tok in tokens:
            pos = t_lower.find(tok)
            if pos >= 0:
                break
    if pos < 0:
        return (text[:window] + "...") if len(text) > window else text

    start = max(0, pos - window)
    end = min(len(text), pos + window)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _search(query: str, top_k: int, sources: list[str] | None, refresh: bool) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []

    allowed = {"local", "global", "prior", "methodology"}
    if sources:
        scope = [s for s in sources if s in allowed]
        if not scope:
            scope = list(allowed)
    else:
        scope = list(allowed)

    index = _build_disk_index(force=bool(refresh))

    q_lower = q.lower()
    tokens = _tokenize_query(q)

    scored = []
    for item in index:
        if item.get("source") not in scope:
            continue
        text_lower = item.get("text_lower") or ""
        if not text_lower:
            continue
        phrase = text_lower.count(q_lower) if q_lower else 0
        token_hits = sum(text_lower.count(t) for t in tokens) if tokens else 0
        score = phrase * 10 + token_hits
        if score <= 0:
            continue
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, item in scored[: max(1, top_k)]:
        title = item.get("title") or ""
        path = item.get("path") or ""
        excerpt = _make_excerpt(item.get("text") or "", q_lower, tokens)
        out.append(f"[{item.get('source')}] {title}\n[path] {path}\n[score] {score}\n{excerpt}")
    return out

