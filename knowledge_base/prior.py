from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF


# -----------------------------
# Config / schema
# -----------------------------

PASAX_KEYS = [
    "title",
    "arxiv_id",
    "depth",
    "child",
    "abstract",
    "sections",
    "source",
    "extra",
    "select_score",
    "comment",
    "authors",
    "citations",
    "h5",
    "IF",
    "CCF",
    "output",
    "execution_time",
]


# Name used by searchkb / downstream indexing
DEFAULT_SOURCE = "globalprior"


# -----------------------------
# Utilities
# -----------------------------

_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def _norm_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def stable_id(*parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def make_pasax_entry(
    title: str,
    abstract: str,
    source: str,
    extra: Optional[Dict[str, Any]] = None,
    select_score: int = 0,
) -> Dict[str, Any]:
    extra = extra or {}
    entry: Dict[str, Any] = {
        "title": title,
        "arxiv_id": "",
        "depth": 0,
        "child": {},
        "abstract": abstract,
        "sections": "",
        "source": source,
        "extra": extra,
        "select_score": select_score,
        "comment": "",
        "authors": [],
        "citations": [],
        "h5": "",
        "IF": "",
        "CCF": "",
        "output": abstract,
        "execution_time": 0,
    }
    for k in PASAX_KEYS:
        if k not in entry:
            if k in ("child", "extra"):
                entry[k] = {}
            elif k in ("authors", "citations"):
                entry[k] = []
            else:
                entry[k] = ""
    return entry


# -----------------------------
# Symbol normalization
# -----------------------------

# Common PUA fixes for PPT-exported PDFs (Symbol/MT Extra glyph mapping).
# This is intentionally conservative; we avoid mapping ambiguous glyphs.
PUA_MAP: Dict[str, str] = {
    "\uf02b": "+",
    "\uf02d": "-",
    "\uf0b1": "±",
    "\uf03d": "=",
    "\uf0a3": "→",  # sometimes right arrow
    "\uf0de": "⇒",
    "\uf0ce": "⇔",
    "\uf0d7": "×",
    "\uf0b7": "·",
    "\uf0a5": "∞",
    "\uf0b4": "′",
    "\uf0b0": "°",
    "\uf028": "(",
    "\uf029": ")",
    "\uf05b": "[",
    "\uf05d": "]",
    "\uf07b": "{",
    "\uf07d": "}",
    "\uf0a8": "∈",
    "\uf0c6": "∝",
    "\uf0b2": "≥",
    "\uf0b3": "≤",
    "\uf0a1": "∑",
    "\uf0d1": "∫",
    "\uf0ac": "¬",
    "\uf0ad": "∧",
    "\uf0ae": "∨",
    "\uf0b9": "≠",
    "\uf0bb": "≈",
    "\uf0bc": "≡",
    "\uf0bd": "≤",
    "\uf0be": "≥",
    "\uf061": "α",  # Symbol font: a -> alpha (often appears as U+F061)
    "\uf071": "θ",  # Symbol font: q -> theta (often appears as U+F071)
    "\uf07e": "∼",  # approx / similar (often appears as U+F07E)
}

# Vector arrow in some PDFs appears as a PUA glyph near \uf072 or \uf0de; we handle robustly with regex.
_VEC_PUA = re.compile(r"([A-Za-z])[\uf072\uf076\uf0ae\uf0af]")

# Unicode greek -> LaTeX (keep minimal set, extendable)
GREEK_LATEX = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
}

MATH_SYM_LATEX = {
    "→": r"\to",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "×": r"\times",
    "·": r"\cdot",
    "∈": r"\in",
    "∝": r"\propto",
    "∞": r"\infty",
    "≤": r"\le",
    "≥": r"\ge",
    "≠": r"\ne",
    "≈": r"\approx",
    "≡": r"\equiv",
    "∑": r"\sum",
    "∫": r"\int",
}


# Allowed ranges: we treat these as safe and never "unmap".
# We must include CJK punctuation and combining diacritics (for arrows like \u20d7).
_ALLOWED_RANGES = [
    (0x0009, 0x000D),  # tabs/newlines
    (0x0020, 0x007E),  # ASCII printable
    (0x00A0, 0x00FF),  # Latin-1
    (0x0300, 0x036F),  # Combining Diacritical Marks (avoid isolating accents)
    (0x0100, 0x024F),  # Latin Extended
    (0x0370, 0x03FF),  # Greek
    (0x2000, 0x206F),  # General punctuation
    (0x20A0, 0x20CF),  # Currency
    (0x20D0, 0x20FF),  # Combining Diacritical Marks for Symbols (incl. vector arrow)
    (0x2100, 0x214F),  # Letterlike symbols
    (0x2150, 0x218F),  # Number forms
    (0x2190, 0x21FF),  # Arrows
    (0x2200, 0x22FF),  # Math operators
    (0x2300, 0x23FF),  # Misc technical
    (0x25A0, 0x25FF),  # Geometric shapes
    (0x2700, 0x27BF),  # Dingbats
    (0x3000, 0x303F),  # CJK Symbols & punctuation
    (0x3040, 0x30FF),  # Hiragana/Katakana
    (0x3400, 0x4DBF),  # CJK Ext A
    (0x4E00, 0x9FFF),  # CJK Unified
    (0xF900, 0xFAFF),  # CJK Compatibility ideographs
    (0xFF00, 0xFFEF),  # Halfwidth/fullwidth
]

def _is_allowed_char(ch: str) -> bool:
    if ch == "\n":
        return True
    o = ord(ch)
    for a, b in _ALLOWED_RANGES:
        if a <= o <= b:
            return True
    # Drop explicit surrogates
    if 0xD800 <= o <= 0xDFFF:
        return False
    # Treat other combining marks as safe
    if unicodedata.category(ch).startswith("M"):
        return True
    return False


def normalize_text(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize a text string:
    - NFKC normalization (stabilize fullwidth and compatibility glyphs)
    - Repair common PUA glyphs
    - Repair vector arrow PUA patterns
    - Isolate disallowed/unmapped chars with reversible token "⟦U+XXXX⟧"
    Returns (normalized_text, stats)
    """
    stats: Dict[str, Any] = {
        "len_in": len(text),
        "pua_replaced": 0,
        "vec_pua_fixed": 0,
        "unmapped_isolated": 0,
    }

    # Step 0: normalize whitespace lightly (keep newlines)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 1: NFKC
    text = unicodedata.normalize("NFKC", text)

    # Step 2: PUA replacement
    if any(ch in text for ch in PUA_MAP):
        out = []
        for ch in text:
            if ch in PUA_MAP:
                out.append(PUA_MAP[ch])
                stats["pua_replaced"] += 1
            else:
                out.append(ch)
        text = "".join(out)

    # Step 3: vector arrow PUA (very common: v)
    def _vec_sub(m: re.Match) -> str:
        stats["vec_pua_fixed"] += 1
        return m.group(1) + "\u20d7"  # combining vector arrow above
    text = _VEC_PUA.sub(_vec_sub, text)

    # Step 4: isolate unmapped characters (but keep safe ranges)
    out2 = []
    for ch in text:
        if _is_allowed_char(ch):
            out2.append(ch)
        else:
            stats["unmapped_isolated"] += 1
            out2.append(f"⟦U+{ord(ch):04X}⟧")
    text = "".join(out2)

    # Canonicalize any damaged unknown-glyph tokens (avoid spaces inside: '⟦U + F061⟧')
    text = re.sub(r"⟦U\s*\+\s*([0-9A-Fa-f]{4,6})\s*⟧", lambda m: f"⟦U+{m.group(1).upper()}⟧", text)
    return text, stats


def latexify_math(expr: str) -> str:
    """
    Convert unicode math-ish content to LaTeX-ish (best-effort, no OCR).

    Design goals:
    - Keep readability for RAG/LLM.
    - Be conservative: do NOT over-convert normal text.
    - Preserve our reversible unknown-glyph tokens: ⟦U+XXXX⟧ (never insert spaces inside),
      except for a small set of well-identified vector-arrow tokens (⟦U+F072⟧) that we normalize.
    """
    expr = expr.strip()
    if not expr:
        return expr

    # Prevent accidental LaTeX block markers from being nested.
    expr = expr.replace("\\[", "").replace("\\]", "")
    expr = expr.replace("\\(", "").replace("\\)", "")

    # Remove spacing-separated combining accents (common extraction artefact like 'r ́')
    expr = re.sub(r"\s+[\u0300-\u036F]+", "", expr)

    # Remove repeated leading rho tokens that are almost always decoration/arrow artefacts in PPT-export PDFs.
    # Only apply when it appears at the beginning of an expression.
    expr = re.sub(r"^(?:\s*(?:ρ|\\rho)\s*){2,}", "", expr)

    # Normalize common isolated vector-arrow tokens (from PPT exports) into LaTeX.
    # Typical pattern: ⟦U+F072⟧L or L⟦U+F072⟧ meaning vector L.
    expr = re.sub(r"⟦U\+F072⟧\s*([A-Za-z])", r"\\vec{\1}", expr)
    expr = re.sub(r"([A-Za-z])\s*⟦U\+F072⟧", r"\\vec{\1}", expr)
    expr = expr.replace("⟦U+F072⟧", "")

    # Protect remaining unknown-glyph tokens from later operator-spacing rules.
    unk_tokens = re.findall(r"⟦U\+[0-9A-F]{4,6}⟧", expr)
    repl = {}
    if unk_tokens:
        for k, t in enumerate(unk_tokens):
            key = f"__UNK{k}__"
            repl[key] = t
            expr = expr.replace(t, key)

    # Map greek and operators
    out = []
    for ch in expr:
        if ch in GREEK_LATEX:
            out.append(GREEK_LATEX[ch])
        elif ch in MATH_SYM_LATEX:
            out.append(MATH_SYM_LATEX[ch])
        else:
            out.append(ch)
    s = "".join(out)

    # Vector arrow combining mark -> \vec{...}
    s = re.sub(r"([A-Za-z])\u20d7", r"\\vec{\1}", s)

    # Normalize spaces lightly around operators (avoid destroying tokens/macros)
    s = re.sub(r"\s+", " ", s)
    # spaces around simple operators, but keep backslash-macros intact
    s = re.sub(r"(?<!\\)\s*([=+\-*/<>])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Restore unknown tokens
    if repl:
        for key, val in repl.items():
            s = s.replace(key, val)

    # Heuristic: common slide shorthand like "T2/a3" -> "T^{2}/a^{3}" (only when equation-ish).
    # Also recover common index notation like v0, r1 in equation-ish contexts.
    if re.search(r"[=/]", s):
        # exponents (2,3,4) are very common in mechanics formulas
        s = re.sub(r"\b([A-Za-z])\s*([234])\b", r"\1^{\2}", s)
        # subscripts (0,1) often denote initial/final or components
        s = re.sub(r"\b([A-Za-z])\s*([01])\b", r"\1_{\2}", s)

    return s

# -----------------------------
# Layout-aware reconstruction
# -----------------------------

@dataclass
class Token:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)


def _iter_tokens_from_rawdict(rd: Dict[str, Any]) -> List[Token]:
    toks: List[Token] = []
    blocks = rd.get("blocks", [])
    for b in blocks:
        if b.get("type", 0) != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "")
                if not t:
                    continue
                bbox = span.get("bbox", [0, 0, 0, 0])
                x0, y0, x1, y1 = bbox
                toks.append(
                    Token(
                        text=t,
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        size=float(span.get("size", 0.0) or 0.0),
                        font=str(span.get("font", "") or ""),
                    )
                )
    return toks


def _cluster_lines(tokens: List[Token], y_tol: float = 2.5) -> List[List[Token]]:
    """
    Group tokens into lines by y (using token center).
    """
    if not tokens:
        return []
    tokens = sorted(tokens, key=lambda t: (t.cy, t.cx))
    lines: List[List[Token]] = []
    cur: List[Token] = [tokens[0]]
    cur_y = tokens[0].cy
    for tok in tokens[1:]:
        if abs(tok.cy - cur_y) <= y_tol:
            cur.append(tok)
            cur_y = (cur_y * (len(cur) - 1) + tok.cy) / len(cur)
        else:
            lines.append(sorted(cur, key=lambda t: t.x0))
            cur = [tok]
            cur_y = tok.cy
    lines.append(sorted(cur, key=lambda t: t.x0))
    return lines


def _merge_tokens_into_text(tokens: List[Token]) -> str:
    """
    Merge tokens in a line; add spaces if gaps are large.
    Also attempt superscript/subscript using token sizes and y positions.
    """
    if not tokens:
        return ""

    sizes = [t.size for t in tokens if t.size > 0]
    med_size = statistics.median(sizes) if sizes else 0.0
    ys0 = [t.y0 for t in tokens]
    ys1 = [t.y1 for t in tokens]
    med_y0 = statistics.median(ys0)
    med_y1 = statistics.median(ys1)

    def is_super(t: Token) -> bool:
        if med_size <= 0:
            return False
        if t.size <= 0:
            return False
        if t.size < 0.82 * med_size and (t.y1 < med_y1 - 0.18 * med_size):
            return True
        return False

    def is_sub(t: Token) -> bool:
        if med_size <= 0:
            return False
        if t.size <= 0:
            return False
        if t.size < 0.82 * med_size and (t.y0 > med_y0 + 0.18 * med_size):
            return True
        return False

    out: List[str] = []
    prev_x1 = None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        gap = 0.0 if prev_x1 is None else max(0.0, t.x0 - prev_x1)
        if prev_x1 is not None and gap > 2.0:  # heuristic
            out.append(" ")

        if is_super(t) or is_sub(t):
            # group contiguous super/sub tokens
            kind_super = is_super(t)
            group = [t.text]
            j = i + 1
            while j < len(tokens) and (is_super(tokens[j]) if kind_super else is_sub(tokens[j])):
                # if too far in x, stop
                if tokens[j].x0 - tokens[j-1].x1 > 6.0:
                    break
                group.append(tokens[j].text)
                j += 1
            gtxt = _norm_ws("".join(group))
            gtxt, _ = normalize_text(gtxt)
            gtxt = latexify_math(gtxt)
            if not out:
                # no base token on the left; keep as normal text
                out.append(gtxt)
            else:
                if kind_super:
                    out.append(f"^{{{gtxt}}}")
                else:
                    out.append(f"_{{{gtxt}}}")
            prev_x1 = tokens[j-1].x1
            i = j
            continue

        seg = t.text
        seg, _ = normalize_text(seg)
        out.append(seg)
        prev_x1 = t.x1
        i += 1

    return _norm_ws("".join(out))


def extract_page_text_pymupdf_layout(page: fitz.Page) -> str:
    """
    Best-effort: use rawdict + line clustering to reconstruct text.
    """
    try:
        rd = page.get_text("rawdict")
        tokens = _iter_tokens_from_rawdict(rd)
        # If tokens are too few, fallback
        if len(tokens) < 3:
            return _norm_ws(page.get_text("text", sort=True) or "")
        lines = _cluster_lines(tokens)

        # Precompute per-line bbox and raw text for fraction reconstruction.
        def _line_bbox(ln: List[Token]) -> Tuple[float, float, float, float]:
            x0 = min(t.x0 for t in ln)
            y0 = min(t.y0 for t in ln)
            x1 = max(t.x1 for t in ln)
            y1 = max(t.y1 for t in ln)
            return x0, y0, x1, y1

        def _is_frac_bar_line(text: str) -> bool:
            s = text.strip()
            if not s:
                return False
            # many PDFs export fraction bar as repeated dashes/underscores
            if re.fullmatch(r"[\-\–\—\_\u2500\u2501\u2010\u2011\u2212]{2,}", s):
                return True
            return False

        line_texts = [_merge_tokens_into_text(ln) for ln in lines]
        line_bboxes = [_line_bbox(ln) for ln in lines]

        # Reconstruct very clear fractions: (numerator) / (denominator) stacked with a bar line.
        consumed = set()
        rebuilt: List[str] = []
        for idx, lt in enumerate(line_texts):
            if idx in consumed:
                continue
            if _is_frac_bar_line(lt):
                # find nearest non-empty above and below
                up = idx - 1
                while up >= 0 and not line_texts[up].strip():
                    up -= 1
                dn = idx + 1
                while dn < len(line_texts) and not line_texts[dn].strip():
                    dn += 1
                if up >= 0 and dn < len(line_texts):
                    bx0, by0, bx1, by1 = line_bboxes[idx]
                    ux0, uy0, ux1, uy1 = line_bboxes[up]
                    dx0, dy0, dx1, dy1 = line_bboxes[dn]
                    # x-overlap ratio with bar
                    ov_up = max(0.0, min(bx1, ux1) - max(bx0, ux0)) / max(1e-6, (bx1 - bx0))
                    ov_dn = max(0.0, min(bx1, dx1) - max(bx0, dx0)) / max(1e-6, (bx1 - bx0))
                    if ov_up > 0.55 and ov_dn > 0.55:
                        num = latexify_math(line_texts[up])
                        den = latexify_math(line_texts[dn])
                        rebuilt.append(f"\\frac{{{num}}}{{{den}}}")
                        consumed.update({up, idx, dn})
                        continue
            rebuilt.append(lt)
        line_texts = rebuilt
        # join: infer paragraph breaks by y gaps is hard; do simple heuristic:
        # split when a line is empty or when it looks like a heading.
        buf: List[str] = []
        last = None
        for lt in line_texts:
            lt = lt.strip()
            if not lt:
                if buf and buf[-1] != "":
                    buf.append("")
                last = lt
                continue
            if _looks_like_heading(lt):
                if buf and buf[-1] != "":
                    buf.append("")
                buf.append(lt)
                buf.append("")
            else:
                buf.append(lt)
            last = lt
        return _norm_ws("\n".join(buf))
    except Exception:
        return _norm_ws(page.get_text("text", sort=True) or "")


def extract_page_text_pymupdf_simple(page: fitz.Page) -> str:
    return _norm_ws(page.get_text("text", sort=True) or "")


def extract_text_pdfminer(pdf_path: Path) -> Optional[List[str]]:
    """
    Optional: pdfminer.six extraction per-page.
    Returns list of page texts, or None if pdfminer not available.
    """
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return None

    try:
        # Extract all; split by form feed '\f' (pdfminer page breaks)
        all_text = extract_text(str(pdf_path))
        pages = all_text.split("\f")
        # Trim trailing empty
        pages = [p.strip() for p in pages if p.strip()]
        return pages if pages else None
    except Exception:
        return None


def extract_text_pdftotext(pdf_path: Path) -> Optional[List[str]]:
    """
    Optional: poppler pdftotext extraction (-layout). Returns per-page list.
    Requires 'pdftotext' binary.
    """
    if shutil.which("pdftotext") is None:
        return None
    try:
        # -layout keeps columns; -enc UTF-8 for safety
        cmd = ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if p.returncode != 0:
            return None
        s = p.stdout.decode("utf-8", errors="ignore")
        pages = s.split("\f")
        # Keep page alignment: do not drop empty pages; strip each page only.
        pages = [p.strip() for p in pages]
        # Remove a single trailing empty page if it is an artifact.
        if pages and pages[-1] == "":
            pages = pages[:-1]
        return pages if pages else None
    except Exception:
        return None


# -----------------------------
# Quality scoring / selection
# -----------------------------

_PUA_CHAR_RE = re.compile(r"[\uf000-\uf8ff]")
_UNK_TOKEN_RE = re.compile(r"⟦U\+[0-9A-F]{4,6}⟧")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_GARBAGE_FRACTION_RE = re.compile(r"[÷øöçèæ]")
_MATH_CLUE_RE = re.compile(r"[=+\-*/^_<>∈∝∞≤≥≠≈≡∑∫→⇒⇔×·]|[α-ωΑ-Ω]")
_REPL_CHAR_RE = re.compile("\uFFFD")  # replacement character

def _estimate_is_cjk(text: str) -> bool:
    return len(_CJK_RE.findall(text)) >= 20


def quality_score(text: str, prefer_cjk: bool) -> float:
    """
    Lower is better.
    """
    if not text:
        return 1e9
    n = max(1, len(text))
    pua = len(_PUA_CHAR_RE.findall(text)) / n
    unk = len(_UNK_TOKEN_RE.findall(text)) / n
    garbage = len(_GARBAGE_FRACTION_RE.findall(text)) / n
    repl = len(_REPL_CHAR_RE.findall(text)) / n
    math_clue = len(_MATH_CLUE_RE.findall(text))

    # Control chars (excluding newline)
    ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in ("\n", "\t")) / n

    # If CJK preferred, penalize too-low CJK ratio slightly
    cjk_ratio = len(_CJK_RE.findall(text)) / n
    cjk_pen = 0.0
    if prefer_cjk and cjk_ratio < 0.02:
        cjk_pen = 0.1

    # Over-fragmentation: too many very short lines
    lines = [ln.strip() for ln in text.splitlines()]
    short_lines = sum(1 for ln in lines if 0 < len(ln) <= 2)
    frag = short_lines / max(1, len(lines))

    return (
        2.0 * pua
        + 3.0 * unk
        + 4.0 * repl
        + 2.0 * garbage
        + 5.0 * ctrl
        + 0.6 * frag
        + cjk_pen
        - 0.05 * math.log(1 + len(text))  # longer (non-empty) texts often better
        - 0.03 * math.log(1 + math_clue)
    )


# -----------------------------
# Math detection / grouping
# -----------------------------

_MATH_CHARS_RE = re.compile(r"[=+\-*/^_<>∈∝∞≤≥≠≈≡∑∫→⇒⇔×·]")
_LATEX_LIKE_RE = re.compile(r"(\\[a-zA-Z]+|[_^]\{)")
_NUM_RE = re.compile(r"\d")
_PUNCT_ONLY_RE = re.compile(r"^[=+\-*/()\[\]{}<>\s]+$")

def _looks_like_heading(line: str) -> bool:
    # Heuristic: short and ends with ":" or contains "第..节/章" or is all caps
    s = line.strip()
    if not s:
        return False
    if len(s) <= 18 and (s.endswith(":") or s.endswith("：")):
        return True
    if re.search(r"第\s*\d+\s*[章节课]", s):
        return True
    if len(s) <= 25 and s.upper() == s and re.search(r"[A-Z]", s):
        return True
    return False


def _math_strength(line: str) -> int:
    """
    Return 0/1/2 for (not math) / (weak math fragment) / (strong math line).
    Weak fragments will be absorbed into a math block if adjacent to strong lines.
    """
    s = line.strip()
    if not s:
        return 0
    if _looks_like_heading(s):
        return 0

    # Our unknown-glyph tokens almost always appear in formulas.
    if "⟦U+" in s:
        return 2

    # Explicit LaTeX-ish markers
    if _LATEX_LIKE_RE.search(s):
        return 2

    op = len(_MATH_CHARS_RE.findall(s))
    dig = len(_NUM_RE.findall(s))
    cjk = len(_CJK_RE.findall(s))

    # Pure operator/punctuation line: treat as weak (avoid starting a display block with just "=").
    if _PUNCT_ONLY_RE.fullmatch(s):
        return 1

    # Common math functions/keywords
    if re.search(r"\b(sin|cos|tan|cot|sec|csc|log|ln|exp|lim|max|min|det|tr)\b", s, flags=re.I):
        return 2

    # Slide-export artefact: variable followed by a digit (often exponent/index fragments like 'T 2', 'a 3')
    has_var_digit = bool(re.search(r"[A-Za-z]\s*\d", s))

    # Character-level "mathish" ratio
    ns = re.sub(r"\s+", "", s)
    if not ns:
        return 0
    mathish = 0
    mathish += len(re.findall(r"[0-9A-Za-z]", ns))
    mathish += 2 * len(_MATH_CHARS_RE.findall(ns))
    mathish += 2 * len(re.findall(r"[α-ωΑ-Ω]", ns))
    ratio = mathish / max(1, len(ns))

    # If there is CJK content, require stronger math evidence.
    if cjk > 0 and (op + dig) < 2 and ratio < 0.60 and "=" not in s:
        return 0

    # Equation-like lines
    if "=" in s and (op + dig) >= 1:
        return 2

    # Strong operator density
    if op >= 2:
        return 2

    # Digit-heavy with little CJK
    if dig >= 4 and cjk == 0:
        return 2

    # Strong mathish ratio
    if ratio >= 0.72 and cjk <= 1:
        return 2
    if ratio >= 0.62 and cjk == 0 and (op + dig) >= 1:
        return 2

    # Weak cases: variable-digit fragment (often exponent/index), standalone numbers, short math-ish tokens
    if has_var_digit and len(s) <= 40:
        return 1
    if s.isdigit():
        return 1
    if len(s) <= 24 and (op >= 1 or dig >= 2):
        return 1

    return 0


def _clean_garbage_lines(lines: List[str]) -> List[str]:
    """
    Remove lines that are almost surely layout garbage (e.g., only ÷øöçèæ).
    """
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        # Drop pure garbage lines
        if re.fullmatch(r"[÷øöçèæ\s]+", s):
            continue
        # Drop lines dominated by known fraction-layout garbage (pdftotext artefacts)
        g = len(_GARBAGE_FRACTION_RE.findall(s))
        if g >= 6 and g / max(1, len(s)) > 0.45:
            continue
        out.append(ln)
    return out


def group_into_segments(page_text: str) -> List[Tuple[str, str]]:
    """
    Split a page into segments: ("text", ...) or ("display_math", ...).

    Key behaviors:
    - Consecutive math lines are grouped into a single display-math block.
    - "Weak" math fragments (e.g., lone numbers, punctuation-only lines) are absorbed into math blocks
      only when adjacent to strong math lines, preventing junk like "\\[=\\]" from being produced alone.
    - Allows up to 2 blank lines *inside* a math block (PPT exports often insert blank lines between tokens).
    """
    lines = page_text.splitlines()
    lines = _clean_garbage_lines(lines)

    # First pass: compute strengths for non-empty lines
    strength: List[int] = []
    nonempty_idx: List[int] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            strength.append(0)
            continue
        st = _math_strength(s)
        strength.append(st)
        nonempty_idx.append(i)

    # Second pass: promote weak fragments if adjacent to a strong math line
    for i in nonempty_idx:
        if strength[i] != 1:
            continue
        # find previous non-empty
        prev_i = None
        for j in range(i - 1, -1, -1):
            if lines[j].strip():
                prev_i = j
                break
        next_i = None
        for j in range(i + 1, len(lines)):
            if lines[j].strip():
                next_i = j
                break
        if (prev_i is not None and strength[prev_i] >= 2) or (next_i is not None and strength[next_i] >= 2):
            strength[i] = 2

    segs: List[Tuple[str, str]] = []
    buf: List[str] = []
    mode: Optional[str] = None  # "text" or "math"
    blank_run = 0

    def flush():
        nonlocal buf, mode, blank_run
        if not buf:
            mode = None
            blank_run = 0
            return

        if mode == "math":
            # Build math lines; drop purely-punctuation lines *unless* the whole block would become empty.
            raw_mlines = [ln.strip() for ln in buf if ln.strip()]
            mlines = [ln for ln in raw_mlines if not _PUNCT_ONLY_RE.fullmatch(ln)]
            if not mlines:
                # fallback to original content as text
                content = _norm_ws("\n".join(buf))
                segs.append(("text", content))
            else:
                if len(mlines) == 1:
                    expr = latexify_math(mlines[0])
                else:
                    expr = r" \\ ".join(latexify_math(ln) for ln in mlines)
                segs.append(("display_math", f"\\[\n{expr}\n\\]"))
        else:
            content = _norm_ws("\n".join(buf))
            segs.append(("text", content))

        buf = []
        mode = None
        blank_run = 0

    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            blank_run += 1
            # If in math mode, tolerate up to 2 blank lines as PPT token separators
            if mode == "math" and blank_run <= 2:
                continue
            flush()
            continue

        blank_run = 0
        is_math = strength[i] >= 2

        if mode is None:
            mode = "math" if is_math else "text"
            buf.append(ln)
            continue

        if mode == "math":
            if is_math:
                buf.append(ln)
            else:
                flush()
                mode = "text"
                buf.append(ln)
        else:  # text mode
            # Start a display-math block only if it's reasonably compact or strongly math-like.
            if is_math and len(s) <= 120:
                flush()
                mode = "math"
                buf.append(ln)
            else:
                buf.append(ln)

    flush()
    return segs


# -----------------------------
# Chunking (structure-aware, math-safe)
# -----------------------------

def chunk_segments(
    segments: List[Tuple[str, str]],
    chunk_size: int = 1400,
    overlap: int = 220,
) -> List[str]:
    """
    Chunk a list of segments, never splitting display_math segments.
    Prefer splitting on paragraph boundaries between segments.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush_with_overlap():
        nonlocal cur, cur_len
        if not cur:
            return
        txt = _norm_ws("\n\n".join(cur))
        if txt:
            chunks.append(txt)
        # prepare overlap
        if overlap <= 0:
            cur = []
            cur_len = 0
            return
        # create overlap from the end, but do not cut inside display_math
        tail = txt[-overlap:]
        # If tail cuts a display-math, back off to before "\["
        i = tail.rfind("\\[")
        j = tail.rfind("\\]")
        if i != -1 and (j == -1 or j < i):
            # back off: remove incomplete block
            tail = tail[:i].strip()
        # Overlap is just a repeated hint; remove block markers to avoid unmatched delimiters.
        tail = tail.replace('\\[', '').replace('\\]', '')
        cur = [tail] if tail else []
        cur_len = len(tail)

    for kind, content in segments:
        piece = content.strip()
        if not piece:
            continue
        add_len = len(piece) + (2 if cur else 0)

        if cur_len + add_len <= chunk_size:
            cur.append(piece)
            cur_len += add_len
            continue

        # If current chunk not empty, flush it
        if cur:
            flush_with_overlap()

        # If single segment too large, split only if it's text (never split display_math)
        if len(piece) > chunk_size and kind == "text":
            # split by sentences/paragraphs within the text
            parts = _split_long_text(piece, chunk_size)
            for p in parts:
                if not p.strip():
                    continue
                if len(p) <= chunk_size:
                    if cur_len + len(p) + (2 if cur else 0) > chunk_size and cur:
                        flush_with_overlap()
                    cur.append(p)
                    cur_len += len(p) + (2 if cur else 0)
                else:
                    # hard fallback
                    cur.append(p[:chunk_size])
                    flush_with_overlap()
            continue

        # Otherwise, start new chunk with this segment
        cur.append(piece)
        cur_len = len(piece)

    if cur:
        txt = _norm_ws("\n\n".join(cur))
        if txt:
            chunks.append(txt)
    return chunks


def _split_long_text(text: str, chunk_size: int) -> List[str]:
    """
    Split long text on paragraph or sentence boundaries.
    """
    text = _norm_ws(text)
    if len(text) <= chunk_size:
        return [text]
    parts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        window = text[start:end]
        boundary = None

        m = window.rfind("\n\n")
        if m != -1 and m > 200:
            boundary = start + m

        if boundary is None:
            punct = max(window.rfind("。"), window.rfind("."), window.rfind("!"), window.rfind("?"), window.rfind("；"), window.rfind(";"))
            if punct != -1 and punct > 200:
                boundary = start + punct + 1

        if boundary is None:
            boundary = end

        parts.append(text[start:boundary].strip())
        if boundary >= n:
            break
        start = boundary
    return parts


# -----------------------------
# PDF ingestion
# -----------------------------

@dataclass
class PageChoice:
    page_index: int
    chosen_extractor: str
    score: float
    stats: Dict[str, Any]


def read_pdf_text_super(pdf_path: Path) -> Tuple[str, List[Tuple[int, str]], Dict[str, Any]]:
    """
    Returns:
      - full_text (with [Page i] separators)
      - pages: list of (page_index, cleaned_page_text)
      - info: extraction report
    """
    doc = fitz.open(pdf_path)
    n_pages = len(doc)

    # Optional multi-extractors (whole PDF), per-page fallback if lengths match.
    miner_pages = extract_text_pdfminer(pdf_path)
    poppler_pages = extract_text_pdftotext(pdf_path)

    # Align pdftotext pages to document page count if necessary.
    if poppler_pages is not None and len(poppler_pages) != n_pages and shutil.which('pdftotext') is not None:
        aligned: List[str] = []
        for pi in range(n_pages):
            cmd = ['pdftotext', '-layout', '-enc', 'UTF-8', '-f', str(pi+1), '-l', str(pi+1), str(pdf_path), '-']
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if p.returncode != 0:
                aligned.append('')
            else:
                aligned.append(p.stdout.decode('utf-8', errors='ignore').strip())
        poppler_pages = aligned

    report: Dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "n_pages": n_pages,
        "extractors_available": {
            "pdfminer": bool(miner_pages),
            "pdftotext": bool(poppler_pages),
        },
        "pages": [],
    }

    pages_out: List[Tuple[int, str]] = []
    buf_full: List[str] = []
    prefer_cjk_global = True  # for your corpus this is generally safe; per-page refined below

    for i in range(n_pages):
        page = doc[i]
        candidates: List[Tuple[str, str]] = []  # (name, text)

        t_layout = extract_page_text_pymupdf_layout(page)
        candidates.append(("pymupdf_layout", t_layout))

        t_simple = extract_page_text_pymupdf_simple(page)
        if t_simple and t_simple != t_layout:
            candidates.append(("pymupdf_simple", t_simple))

        if miner_pages and i < len(miner_pages):
            candidates.append(("pdfminer", _norm_ws(miner_pages[i])))

        if poppler_pages and i < len(poppler_pages):
            candidates.append(("pdftotext", _norm_ws(poppler_pages[i])))

        # Normalize + score each
        best = None
        best_score = 1e18
        best_stats: Dict[str, Any] = {}
        prefer_cjk = prefer_cjk_global

        # infer CJK preference on this page from any candidate
        for _, cand in candidates:
            if _estimate_is_cjk(cand):
                prefer_cjk = True
                break

        for name, cand in candidates:
            norm, st = normalize_text(cand)
            # keep line breaks for segmenting
            norm = norm.replace("\n ", "\n")
            sc = quality_score(norm, prefer_cjk=prefer_cjk)
            if sc < best_score:
                best_score = sc
                best = (name, norm)
                best_stats = st

        assert best is not None
        chosen_name, chosen_text = best

        # Keep raw (normalized) page text here; math segmentation is done once later at global assembly
        # to avoid nested "\\[...\\]" artifacts from double-pass segmenting.
        clean_lines = _clean_garbage_lines(chosen_text.splitlines())
        page_final = "\n".join(clean_lines)
        page_final = page_final.replace("\n ", "\n")
        page_final = _MULTI_NL_RE.sub("\n\n", page_final).strip()

        pages_out.append((i, page_final))
        if page_final:
            buf_full.append(f"[Page {i}]\n{page_final}")

        report["pages"].append(
            {
                "page_index": i,
                "chosen_extractor": chosen_name,
                "quality_score": best_score,
                "norm_stats": best_stats,
                "len_page_text": len(page_final),
            }
        )

    doc.close()
    full = _norm_ws("\n\n".join(buf_full))
    return full, pages_out, report


# -----------------------------
# File ingestion
# -----------------------------

def ingest_one_file(
    src_path: Path,
    out_dir: Path,
    chunk_size: int,
    overlap: int,
    write_report: bool = True,
) -> int:
    suffix = src_path.suffix.lower()
    written = 0

    if suffix in [".txt", ".md"]:
        text_raw = src_path.read_text(encoding="utf-8", errors="ignore")
        text, st = normalize_text(text_raw)
        segs = [("text", _norm_ws(text))]
        chunks = chunk_segments(segs, chunk_size=chunk_size, overlap=overlap)

        for i, ck in enumerate(chunks):
            entry = make_pasax_entry(
                title=f"[PRIOR] {src_path.name} :: chunk {i}",
                abstract=ck,
                source=DEFAULT_SOURCE,
                extra={
                    "kind": "prior",
                    "source_file": str(src_path.as_posix()),
                    "chunk_index": i,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "note": "text_file_normalized",
                    "norm_stats": st,
                },
            )
            fid = stable_id(str(src_path), str(i))
            (out_dir / f"PRIOR__{src_path.stem}__{fid}__{i}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1
        return written

    if suffix == ".pdf":
        full_text, pages, report = read_pdf_text_super(src_path)

        # structure-aware segments are built per-page already; for chunking we preserve page boundaries:
        segments: List[Tuple[str, str]] = []
        for pi, ptxt in pages:
            if not ptxt:
                continue
            segments.append(("text", f"[Page {pi}]"))
            # further split this page into segments again to ensure math safety at global level
            segs = group_into_segments(ptxt)
            segments.extend(segs)

        chunks = chunk_segments(segments, chunk_size=chunk_size, overlap=overlap)

        for i, ck in enumerate(chunks):
            entry = make_pasax_entry(
                title=f"[PRIOR] {src_path.name} :: chunk {i}",
                abstract=ck,
                source=DEFAULT_SOURCE,
                extra={
                    "kind": "prior",
                    "source_file": str(src_path.as_posix()),
                    "chunk_index": i,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "note": "pdf_text_super_extracted_no_ocr",
                    "extract_report_summary": {
                        "n_pages": report.get("n_pages", None),
                        "extractors_available": report.get("extractors_available", {}),
                    },
                },
            )
            fid = stable_id(str(src_path), str(i))
            (out_dir / f"PRIOR__{src_path.stem}__{fid}__{i}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1

        if write_report:
            rep_path = out_dir / f"PRIOR__{src_path.stem}__{stable_id(str(src_path),'report')}__report.json"
            rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return written

    return 0


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="knowledge_source/prior")
    ap.add_argument("--output_dir", type=str, default="global_prior")
    ap.add_argument("--chunk_size", type=int, default=1400)
    ap.add_argument("--overlap", type=int, default=220)
    ap.add_argument("--clean", action="store_true", help="清空 output_dir 后再写入")
    ap.add_argument("--no_report", action="store_true", help="不输出每份 PDF 的 report.json")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent  # expect to be knowledge_base/
    in_dir = (project_root / args.input_dir).resolve()
    out_dir = (project_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for p in out_dir.glob("PRIOR__*.json"):
            p.unlink()

    if not in_dir.exists():
        print(f"[prior_super_ultimate_v6.py] input_dir not found: {in_dir}")
        return

    src_files: List[Path] = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        src_files.extend(in_dir.rglob(ext))

    total_written = 0
    for f in sorted(src_files):
        n = ingest_one_file(f, out_dir, args.chunk_size, args.overlap, write_report=not args.no_report)
        total_written += n
        if n > 0:
            print(f"[prior_super_ultimate_v6.py] ingested {f} -> {n} chunks")

    print(f"[prior_super_ultimate_v6.py] DONE. total_chunks={total_written}, output_dir={out_dir}")


if __name__ == "__main__":
    main()
