import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

_INDEX_CACHE: Dict[str, List[dict]] = {}


def _kb_roots() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    landau_root = repo_root / "LANDAU"
    return {
        "local": landau_root / "local_library",
        "global": landau_root / "global_library",
        "prior": landau_root / "global_prior",
        "methodology": landau_root / "global_methodology",
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


def _index_json_dir(root: Path, source: str) -> list[dict]:
    entries: list[dict] = []
    if not root or not root.exists():
        return entries

    for fp in _iter_files(root, (".json",)):
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # If recall/top papers exist, index those; else index the object itself.
        candidates = []
        if isinstance(data, dict):
            extra = data.get("extra") or {}
            if isinstance(extra, dict):
                rp = extra.get("recall_papers") or extra.get("top_papers")
                if isinstance(rp, list):
                    candidates = [c for c in rp if isinstance(c, dict)]
        if not candidates and isinstance(data, list):
            candidates = [c for c in data if isinstance(c, dict)]
        if not candidates and isinstance(data, dict):
            candidates = [data]

        for i, payload in enumerate(candidates[:200]):
            title = payload.get("title", "")
            text = _extract_search_text(payload)
            if not text and title:
                text = title
            entries.append(
                {
                    "source": source,
                    "path": f"{fp.as_posix()}#cand{i}" if len(candidates) > 1 else fp.as_posix(),
                    "title": title,
                    "text": text,
                    "text_lower": (text or "").lower(),
                }
            )

    return entries


def _index_text_dir(root: Path, source: str) -> list[dict]:
    entries: list[dict] = []
    if not root or not root.exists():
        return entries

    for fp in _iter_files(root, (".md", ".txt", ".json")):
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

        entries.append(
            {
                "source": source,
                "path": fp.as_posix(),
                "title": title,
                "text": text,
                "text_lower": (text or "").lower(),
            }
        )

    return entries


def _build_index(kind: str, *, refresh: bool = False) -> list[dict]:
    if refresh:
        _INDEX_CACHE.pop(kind, None)
    if not refresh and kind in _INDEX_CACHE:
        return _INDEX_CACHE[kind]

    roots = _kb_roots()
    index: list[dict] = []

    if kind == "library":
        for src in ("local", "global"):
            index.extend(_index_json_dir(roots.get(src), src))
    elif kind == "methodology":
        index.extend(_index_text_dir(roots.get("methodology"), "methodology"))
    elif kind == "prior":
        index.extend(_index_json_dir(roots.get("prior"), "prior"))

    _INDEX_CACHE[kind] = index
    return index


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


def _rank_results(index: list[dict], query: str, top_k: int) -> list[dict]:
    q = (query or "").strip()
    if not q or not index:
        return []

    try:
        top_k = max(1, int(top_k))
    except Exception:
        top_k = 5

    q_lower = q.lower()
    tokens = _tokenize_query(q)

    scored = []
    for item in index:
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
    hits: list[dict] = []
    for score, item in scored[:top_k]:
        hits.append(
            {
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "path": item.get("path", ""),
                "score": score,
                "excerpt": _make_excerpt(item.get("text") or "", q_lower, tokens),
            }
        )
    return hits


def _format_hits(label: str, query: str, hits: list[dict]) -> str:
    if not query:
        return f"[{label}] Empty query."
    if not hits:
        return f"[{label}] No entries found for query: {query}"

    lines = [f"[{label}] Search results for: {query}"]
    for i, h in enumerate(hits, 1):
        lines.append(
            f"{i}. [{h.get('source')}] {h.get('title', '')}\n"
            f"[path] {h.get('path', '')}\n"
            f"[score] {h.get('score')}\n"
            f"{h.get('excerpt', '')}"
        )
    return "\n".join(lines)


def library_search(query: str, top_k: int = 5, sources: list[str] | None = None, refresh: bool = False) -> str:
    """
    Search across local/global library JSON entries (recall_papers/top_papers aware).
    """
    if refresh:
        _INDEX_CACHE.pop("library", None)
    index = _build_index("library", refresh=refresh)

    allowed = {"local", "global"}
    scope = [s for s in (sources or []) if s in allowed]
    if scope:
        index = [i for i in index if i.get("source") in scope]

    hits = _rank_results(index, query, top_k)
    return _format_hits("Library KB", query, hits)


def methodology_search(query: str, top_k: int = 5, refresh: bool = False) -> str:
    """
    Search methodology notes (.md/.txt/.json) only.
    """
    if refresh:
        _INDEX_CACHE.pop("methodology", None)
    index = _build_index("methodology", refresh=refresh)
    hits = _rank_results(index, query, top_k)
    return _format_hits("Methodology KB", query, hits)


def prior_search(query: str, top_k: int = 5, refresh: bool = False) -> str:
    """
    Search prior outputs only.
    """
    if refresh:
        _INDEX_CACHE.pop("prior", None)
    index = _build_index("prior", refresh=refresh)
    hits = _rank_results(index, query, top_k)
    return _format_hits("Prior KB", query, hits)
