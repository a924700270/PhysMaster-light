from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# ----------------------------
# Helpers
# ----------------------------

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_stem(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return "untitled"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", s)
    return s[:120] if len(s) > 120 else s


def _read_text(fp: Path) -> str:
    # tolerate weird encodings in notes
    return fp.read_text(encoding="utf-8", errors="ignore")


def _file_sha1(fp: Path) -> str:
    h = hashlib.sha1()
    with fp.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _detect_title(text: str, fallback: str) -> str:
    """
    Title heuristic:
    - first markdown H1 '# ...'
    - else first non-empty line (short)
    - else fallback stem
    """
    if not text:
        return fallback
    for line in text.splitlines()[:30]:
        t = line.strip()
        if not t:
            continue
        if t.startswith("#"):
            t = t.lstrip("#").strip()
            if t:
                return t[:200]
        # non-header but short line -> title candidate
        if len(t) <= 80:
            return t
        break
    return fallback


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start_char, end_char, chunk_text).
    Chunking is char-based and deterministic. Good enough for retrieval and avoids extra deps.
    """
    if chunk_size <= 0:
        chunk_size = 1800
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    t = (text or "").strip()
    if not t:
        return []

    chunks = []
    n = len(t)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)

        # try not to cut in the middle of a line
        if end < n:
            cut = t.rfind("\n", start, end)
            if cut > start + int(chunk_size * 0.6):
                end = cut

        chunk = t[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rel_to_kb_root(fp: Path, kb_root: Path) -> str:
    try:
        return fp.resolve().relative_to(kb_root.resolve()).as_posix()
    except Exception:
        return fp.as_posix()


# ----------------------------
# Optional LLM summary
# ----------------------------

def _llm_summarize(text: str, title: str) -> str:
    """
    Optional: summarize with your project LLM utility if available.
    Kept isolated so the script works even if utils are not importable here.
    """
    # NOTE: Don't assume this exists; try import at runtime.
    try:
        from utils.gpt5_utils import call_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"LLM summarization requested but cannot import utils.gpt5_utils.call_model: {e}")

    system_prompt = (
        "You are an expert research engineer. Summarize the following methodology note for future reuse.\n"
        "Output concise, structured Markdown with sections:\n"
        "- Problem pattern\n"
        "- Constraints / pitfalls\n"
        "- Procedure (checklist)\n"
        "- Verification (tests)\n"
        "- Failure modes\n"
        "Be faithful to the text; do not invent."
    )
    user_prompt = f"Title: {title}\n\nNote:\n{text[:12000]}"
    resp = call_model(system_prompt=system_prompt, user_prompt=user_prompt, model_name="gpt-5")
    return (resp or "").strip()


# ----------------------------
# Main ingest
# ----------------------------

@dataclass
class IngestConfig:
    input_dir: Path
    output_dir: Path
    summary_dir: Path
    chunk_size: int
    overlap: int
    llm_summary: bool


def ingest_one_file(fp: Path, cfg: IngestConfig, kb_root: Path) -> Path:
    raw = _read_text(fp)
    stem = _safe_stem(fp.stem)
    title = _detect_title(raw, fallback=stem)

    sha1 = _file_sha1(fp)
    relpath = _rel_to_kb_root(fp, kb_root)
    created_at = _now_iso()

    chunks = _chunk_text(raw, cfg.chunk_size, cfg.overlap)
    chunk_objs = []
    for i, (s, e, ch) in enumerate(chunks):
        chunk_id = f"{stem}__c{i:04d}"
        chunk_objs.append(
            {
                "chunk_id": chunk_id,
                "start": int(s),
                "end": int(e),
                "text": ch,
            }
        )

    entry = {
        "type": "methodology",
        "source": "manual",
        "title": title,
        "path": relpath,                 # path relative to knowledge_base/
        "file_name": fp.name,
        "sha1": sha1,
        "created_at": created_at,
        # For retrieval: keep lots of raw strings at top-level and in chunks.
        "content": raw,
        "chunks": chunk_objs,
        "extra": {
            "recommended_use": "Reusable process / checklist / debugging playbook",
        },
    }

    out_fp = cfg.output_dir / f"{stem}.json"
    _ensure_dir(cfg.output_dir)
    out_fp.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional LLM summary (to a separate .md file + also index JSON)
    if cfg.llm_summary:
        _ensure_dir(cfg.summary_dir)
        summary_md = _llm_summarize(raw, title=title)
        smd_fp = cfg.summary_dir / f"{stem}.summary.md"
        smd_fp.write_text(summary_md + "\n", encoding="utf-8")

        summary_entry = {
            "type": "methodology_summary",
            "source": "llm",
            "title": f"{title} (summary)",
            "path": _rel_to_kb_root(smd_fp, kb_root),
            "file_name": smd_fp.name,
            "created_at": _now_iso(),
            "content": summary_md,
            "extra": {
                "origin_note": relpath,
                "origin_sha1": sha1,
            },
        }
        (cfg.summary_dir / f"{stem}.summary.json").write_text(
            json.dumps(summary_entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return out_fp


def _pick_input_dir(kb_root: Path, user_input: Optional[str]) -> Path:
    if user_input:
        return Path(user_input)

    # Preferred: methodology
    p1 = kb_root / "knowledge_source" / "methodology"
    if p1.exists():
        return p1

    # Fallback: legacy case folder
    p2 = kb_root / "knowledge_source" / "case"
    return p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=None, help="Default: knowledge_base/knowledge_source/methodology (fallback: case)")
    ap.add_argument("--output_dir", type=str, default=None, help="Default: knowledge_base/global_methodology/manual")
    ap.add_argument("--summary_dir", type=str, default=None, help="Default: knowledge_base/global_methodology/summary")
    ap.add_argument("--chunk_size", type=int, default=1800)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--llm_summary", type=int, default=0, help="0/1. Default off.")
    args = ap.parse_args()

    kb_root = Path(__file__).resolve().parent  # .../knowledge_base

    in_dir = _pick_input_dir(kb_root, args.input_dir)

    out_dir = Path(args.output_dir) if args.output_dir else (kb_root / "global_methodology" / "manual")
    sum_dir = Path(args.summary_dir) if args.summary_dir else (kb_root / "global_methodology" / "summary")

    cfg = IngestConfig(
        input_dir=in_dir,
        output_dir=out_dir,
        summary_dir=sum_dir,
        chunk_size=int(args.chunk_size),
        overlap=int(args.overlap),
        llm_summary=bool(int(args.llm_summary)),
    )

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"[methodology.py] input_dir not found: {cfg.input_dir}")

    _ensure_dir(cfg.output_dir)
    if cfg.llm_summary:
        _ensure_dir(cfg.summary_dir)

    files = []
    for ext in (".md", ".txt"):
        files.extend(sorted(cfg.input_dir.rglob(f"*{ext}")))

    if not files:
        print(f"[methodology.py] No .md/.txt files found under: {cfg.input_dir}")
        print("Nothing to ingest.")
        return

    n_ok = 0
    for fp in files:
        try:
            out_fp = ingest_one_file(fp, cfg, kb_root=kb_root)
            n_ok += 1
            print(f"[ok] {fp} -> {out_fp}")
        except Exception as e:
            print(f"[fail] {fp}: {e}")

    print(f"[methodology.py] Done. Ingested {n_ok}/{len(files)} files.")
    print(f"Output dir: {cfg.output_dir}")
    if cfg.llm_summary:
        print(f"Summary dir: {cfg.summary_dir}")


if __name__ == "__main__":
    main()
