# -*- coding: utf-8 -*-
"""
pdf_keyword_search.py

Generalized keyword-to-section extractor for PDFs.

Given:
  - a PDF file path
  - a list of keywords (e.g., ["IEC 62443", "Firewall", "Backup", "Network Management", "System Hardening"])

Produces:
  {
    "<Keyword>": [
      {
        "section": "<top-level section title if available, else closest higher heading>",
        "sub section": "<most specific section number like 20.2.43 or 23.1.4 when present>",
        "page": "<1-based page number or range if span detected>",
        "text": "<concise passage around the occurrence>"
      },
      ...
    ],
    ...
  }

Approach:
1) Parse each PDF page text (PyMuPDF).
2) Detect headings & numeric section patterns (e.g., "23 Cyber Security", "23.1.4 ...", "20.2.43").
3) For each keyword hit, build a local context window (nearby lines + the latest heading stack).
4) Ask AzureChatOpenAI to (a) confirm the hit and (b) return the best section/subsection labels and a clean excerpt.
   - If the LLM fails or is rate-limited, we fall back to a regex/heuristic-only path.

No argparse; configure inside main().
"""

from __future__ import annotations
import os
import re
import json
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import openai
# PDF parsing
try:
    # Attempt to import Document Intelligence SDK. If unavailable,
    # fallback to PyMuPDF for PDF parsing.
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore
    _HAS_DOC_INTELLIGENCE = True
except Exception:
    DocumentIntelligenceClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore
    _HAS_DOC_INTELLIGENCE = False

import fitz  # PyMuPDF
from dotenv import load_dotenv
from typing import Union
# LLM (LangChain Azure OpenAI)
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ---------- Heuristics & Utilities ----------

# Numeric section patterns:  "23", "23.1", "23.1.4", "20.2.43", etc.
SECTION_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,6})\s+(.+?)\s*$")

# Some docs write like: "23. CYBER SECURITY" or "23 CYBER SECURITY"
SECTION_NUM_PUNCT_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)-]?\s+(.+?)\s*$")

# Inline references like "... in section 23.1.4 ..." (helpful fallback)
INLINE_REF_RE = re.compile(r"\b(\d+(?:\.\d+){1,6})\b")

# Merge hyphenated line breaks: "inter-\nface" -> "interface"
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# Squash excessive whitespace (but keep paragraph breaks)
def normalize_text(text: str) -> str:
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    # Normalize Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove multi-spaces on single lines but preserve line breaks
    text = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n"))
    # Remove duplicate blank lines (keep at most one)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class PageInfo:
    index0: int              # 0-based page index
    text: str                # full normalized text for the page
    lines: List[str]         # text split into lines
    headings: List[Tuple[int, str, str]]  # [(line_idx, section_num_or_empty, title_text)]
    # Example heading tuple:
    #  (12, "23.1.4", "Firewall Requirements")  or (7, "23", "Cyber Security")


def load_pdf_pages(
    pdf_path: str,
    doc_client: Optional["DocumentIntelligenceClient"] = None,
    page_indices: Optional[List[int]] = None
) -> List[PageInfo]:
    """
    Load and parse the pages of a PDF document. If Azure Document Intelligence
    (formerly Document Analysis) is available and configured via environment
    variables, the document will be analyzed using the `prebuilt-layout` model
    to extract lines. Otherwise, fall back to PyMuPDF (fitz) for text extraction.

    Parameters
    ----------
    pdf_path : str
        Local filesystem path to the PDF document.

    Returns
    -------
    List[PageInfo]
        A list of PageInfo objects containing normalized page text, lines and
        detected headings.
    """
    # First attempt to use Azure Document Intelligence if available and
    # credentials are supplied.
    # Use Document Intelligence if a client is provided or can be configured.
    if _HAS_DOC_INTELLIGENCE:
        # If an explicit DocumentIntelligenceClient instance is supplied, use it.
        client = doc_client
        # Otherwise attempt to build a client from environment variables.
        if client is None:
            endpoint = os.environ.get("DOCAI_ENDPOINT")
            key = os.environ.get("DOCAI_KEY")
            if endpoint and key:
                try:
                    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
                except Exception:
                    client = None
        if client:
            try:
                # Convert the subset of pages into the proper range string expected by the API.
                # The Document Intelligence service expects a comma-separated string of
                # 1-based page numbers or ranges (e.g., "1-3,5,7-9") for the `pages` parameter.
                pages_param: Optional[str] = None
                if page_indices is not None:
                    # Build a sorted set of candidate indices and convert to 1-based page numbers.
                    numbers = sorted({i + 1 for i in page_indices})
                    if numbers:
                        # Compress consecutive numbers into ranges (e.g., 1,2,3 => "1-3").
                        ranges: List[str] = []
                        start = prev = numbers[0]
                        for n in numbers[1:]:
                            if n == prev + 1:
                                prev = n
                                continue
                            # Emit the previous range
                            if start == prev:
                                ranges.append(str(start))
                            else:
                                ranges.append(f"{start}-{prev}")
                            start = prev = n
                        # Append the final range
                        if start == prev:
                            ranges.append(str(start))
                        else:
                            ranges.append(f"{start}-{prev}")
                        pages_param = ",".join(ranges)
                # Invoke the prebuilt-layout model on the selected pages.
                with open(pdf_path, "rb") as f:
                    poller = client.begin_analyze_document(
                        "prebuilt-layout",
                        body=f,
                        pages=pages_param,
                    )
                result = poller.result()
                pages: List[PageInfo] = []
                for page in result.pages:
                    # page.page_number is 1-based; convert to 0-based
                    index0 = page.page_number - 1
                    # Gather the text from all lines on this page
                    raw_lines = [line.content for line in page.lines] if page.lines else []
                    # Skip pages that have no recognized text (likely scanned images)
                    if not raw_lines:
                        continue
                    raw_text = "\n".join(raw_lines)
                    norm = normalize_text(raw_text)
                    lines = norm.split("\n")
                    headings = detect_headings(lines)
                    pages.append(PageInfo(index0=index0, text=norm, lines=lines, headings=headings))
                # Ensure pages are sorted by their original index (1-based page number)
                pages.sort(key=lambda p: p.index0)
                return pages
            except Exception as e:
                # If Document Intelligence fails for any reason, log and fall back
                print(f"Document Intelligence parsing failed: {e}. Falling back to PyMuPDF.")

    # Fallback: use PyMuPDF to extract plain text
    doc = fitz.open(pdf_path)
    pages: List[PageInfo] = []
    # Determine which pages to process: all or specific indices
    indices_to_process: List[int]
    if page_indices is not None:
        indices_to_process = [i for i in sorted(set(page_indices)) if 0 <= i < len(doc)]
    else:
        indices_to_process = list(range(len(doc)))
    for i in indices_to_process:
        page = doc[i]
        raw = page.get_text("text")
        norm = normalize_text(raw)
        # Skip pages with no alphanumeric content (likely scanned images)
        if not any(ch.isalnum() for ch in norm):
            continue
        lines = norm.split("\n")
        headings = detect_headings(lines)
        pages.append(PageInfo(index0=i, text=norm, lines=lines, headings=headings))
    doc.close()
    # Ensure pages are sorted by their original index
    pages.sort(key=lambda p: p.index0)
    return pages


def detect_headings(lines: List[str]) -> List[Tuple[int, str, str]]:
    """
    Detect potential headings in a page. This function is robust to headings
    split across two lines. It returns a list of tuples (line_idx, section_number,
    title_text) for each detected heading. A heading may be a single line such
    as ``23 Cyber Security`` or a two-line combination where the first line
    contains only the section number and the following line contains the title.

    Parameters
    ----------
    lines : List[str]
        The list of normalized lines on a page.

    Returns
    -------
    List[Tuple[int, str, str]]
        A list of (line_idx, section_number, title_text) tuples.
    """
    results: List[Tuple[int, str, str]] = []
    skip_next = False
    for idx, ln in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        # Try to match standard numeric heading patterns
        m = SECTION_NUM_RE.match(ln)
        if not m:
            m = SECTION_NUM_PUNCT_RE.match(ln)
        if m:
            sec = m.group(1).strip()
            title = m.group(2).strip()
            # Avoid tiny "table of contents" style leftovers (heuristic)
            if len(title) >= 3 and not title.lower().startswith("table of"):
                results.append((idx, sec, title))
            continue
        # If the line contains only a section number (with optional punctuation),
        # consider the next line as the title if it looks like a heading.
        m2 = re.match(r"^\s*(\d+(?:\.\d+){0,6})[.)-]?\s*$", ln)
        if m2 and idx + 1 < len(lines):
            sec = m2.group(1).strip()
            next_ln = lines[idx + 1].strip()
            # Must contain alphabetic characters and should not start with a number
            # to avoid list items. Also restrict extremely long lines.
            if (next_ln and re.search(r"[A-Za-z]", next_ln) and not re.match(r"^\d", next_ln)
                    and len(next_ln) <= 80):
                results.append((idx, sec, next_ln))
                skip_next = True
    return results


def build_heading_stack(pages: List[PageInfo]) -> Dict[Tuple[int, int], Dict[str, str]]:
    """
    Walk through pages and lines; maintain the closest 'current' section/subsection
    for each line by propagating the last seen heading numbers.
    Returns a mapping (page_index0, line_idx) -> {"section": "...", "sub_section": "..."}.
    """
    anchor_map: Dict[Tuple[int, int], Dict[str, str]] = {}

    current_section = ""
    current_title = ""
    current_subsection = ""

    for p in pages:
        # We will scan lines top->bottom; whenever we see headings, update current_section/subsection
        head_iter = iter(p.headings)
        next_head = next(head_iter, None)

        for li, ln in enumerate(p.lines):
            # Advance heading pointer if we've reached next heading line
            while next_head and li >= next_head[0]:
                _idx, sec_num, title = next_head
                # Decide if it's a top-level or deeper subsection by counting dots
                dot_count = sec_num.count(".") if sec_num else 0
                if dot_count == 0:
                    current_section = f"{sec_num} {title}".strip() if sec_num else title
                    current_subsection = sec_num or ""
                    current_title = title
                else:
                    # deeper subsection
                    current_subsection = sec_num
                    # keep section as the highest-level number + title (best effort)
                    if current_section == "":
                        # If we never saw a top-level, synthesize one from first component
                        top_num = sec_num.split(".")[0]
                        current_section = f"{top_num} {title}".strip()
                        current_title = title
                next_head = next(head_iter, None)

            anchor_map[(p.index0, li)] = {
                "section": current_section or "",
                "sub_section": current_subsection or "",
            }

    return anchor_map

def _as_pattern(pattern_or_kw: Union[str, re.Pattern]) -> re.Pattern:
    """Return a compiled, case-insensitive regex for the given string or pattern.
    Uses your plural-aware compile_keyword_pattern if available."""
    if isinstance(pattern_or_kw, re.Pattern):
        return pattern_or_kw
    # If you already defined compile_keyword_pattern, prefer it:
    try:
        return compile_keyword_pattern(pattern_or_kw)  # plural-aware
    except NameError:
        # Fallback: exact phrase match with word boundaries
        return re.compile(rf"\b{re.escape(str(pattern_or_kw))}\b", re.IGNORECASE)

def compile_keyword_pattern(kw: str) -> re.Pattern:
    """
    Build a case-insensitive regex that matches the keyword phrase,
    allowing plural forms of the last alphabetic word.
    Always returns a compiled regex (re.Pattern).
    """
    m = re.search(r"([A-Za-z]+)(?!.*[A-Za-z])", kw)
    if not m:
        # Keyword has no alphabetic word (e.g., "IEC 62443") → exact phrase
        parts = re.split(r"\s+", kw.strip())
        escaped = r"\s+".join(re.escape(p) for p in parts if p)
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

    last_word = m.group(1)
    start, end = m.span(1)

    def plural_variants(word: str) -> list[str]:
        wl = word.lower()
        variants = {word}  # singular
        if re.search(r"(s|x|z|ch|sh)$", wl):
            variants.add(word + "es")
        elif wl.endswith("y") and len(word) > 1 and word[-2].lower() not in "aeiou":
            variants.add(word[:-1] + "ies")
        elif wl.endswith("f"):
            variants.add(word[:-1] + "ves")
            variants.add(word + "s")
        elif wl.endswith("fe"):
            variants.add(word[:-2] + "ves")
            variants.add(word + "s")
        else:
            variants.add(word + "s")
        return sorted(variants, key=len, reverse=True)

    alts = plural_variants(last_word)
    prefix = kw[:start]
    suffix = kw[end:]

    def esc_with_spaces(s: str) -> str:
        return r"\s+".join(re.escape(p) for p in re.split(r"\s+", s) if p)

    prefix_pat = esc_with_spaces(prefix)
    suffix_pat = esc_with_spaces(suffix)
    alts_pat = "|".join(re.escape(a) for a in alts)

    if prefix_pat and suffix_pat:
        pat = rf"\b{prefix_pat}\s+(?:{alts_pat})\s+{suffix_pat}\b"
    elif prefix_pat:
        pat = rf"\b{prefix_pat}\s+(?:{alts_pat})\b"
    elif suffix_pat:
        pat = rf"\b(?:{alts_pat})\s+{suffix_pat}\b"
    else:
        pat = rf"\b(?:{alts_pat})\b"

    return re.compile(pat, re.IGNORECASE)



def find_keyword_hits(pages: List[PageInfo], keywords: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan through all pages and lines to find occurrences of each keyword. Supports
    multi-word keywords split across adjacent lines (e.g., "IEC" at the end of
    one line and "62443" at the beginning of the next). Content pages near
    the beginning of the document (table of contents) are skipped.

    Parameters
    ----------
    pages : List[PageInfo]
        Parsed pages of the document.
    keywords : List[str]
        Keywords to search for.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A mapping from each keyword to a list of hit dictionaries.
    """
    hits: Dict[str, List[Dict[str, Any]]] = {kw: [] for kw in keywords}
    kw_patterns = {kw: compile_keyword_pattern(kw) for kw in keywords}
    kw_multiword = {kw: len(kw.split()) > 1 for kw in keywords}

    for p in pages:
        # Skip table of contents pages (commonly first one or two pages) if they contain 'contents'
        lower_text = p.text.lower()
        if p.index0 <= 2 and "contents" in lower_text:
            continue
        # Skip pages that have no lines (likely scanned images)
        if not p.lines:
            continue
        for li, ln in enumerate(p.lines):
            for kw, pat in kw_patterns.items():
                found = False
                if pat.search(ln):
                    found = True
                elif kw_multiword[kw] and li + 1 < len(p.lines):
                    # Attempt to match keyword across this line and the next
                    combined = f"{ln} {p.lines[li + 1]}"
                    if pat.search(combined):
                        found = True
                if found:
                    context = get_context_window(p.lines, li, radius=4)
                    hits[kw].append({
                        "page_idx0": p.index0,
                        "line_idx": li,
                        "line": ln,
                        "context": context,
                    })
    return hits


def get_context_window(lines: List[str], center_idx: int, radius: int = 4) -> str:
    lo = max(0, center_idx - radius)
    hi = min(len(lines), center_idx + radius + 1)
    block = "\n".join(lines[lo:hi]).strip()
    return block


def page_span_for_block(page: PageInfo, block: str) -> str:
    """
    Heuristic: returns the page number as a string.
    If desired, this is where you'd detect multi-page spans; for now return 1-based page index.
    """
    return str(page.index0 + 1)


# ---------- LLM Orchestration ----------

SYSTEM_PROMPT = (
    "You are a precise document parser. Your job is to map a keyword occurrence to the most specific "
    "section numbering and a short, faithful excerpt (1-3 sentences) that includes the keyword.\n"
    "Rules:\n"
    "1) Use the MOST SPECIFIC section number visible (e.g., '20.2.43' over '20.2').\n"
    "2) 'section' must be the nearest higher-level title (e.g., '23 Cyber Security').\n"
    "3) 'sub section' must be the exact numbering like '23.1.4' (empty if truly none available).\n"
    "4) Keep 'text' concise and quote only from the provided context. Do not fabricate.\n"
    "5) If multiple plausible numbers appear, choose the one governing the line with the keyword.\n"
    "Return ONLY JSON object with keys: section, sub section, text."
)

def ask_llm_for_hit(
    llm: AzureChatOpenAI,
    keyword: str,
    context_block: str,
    heading_hint: Dict[str, str],
    heading_lines_nearby: str
) -> Optional[Dict[str, str]]:
    """
    Ask the LLM to produce a struct:
      { "section": "...", "sub section": "...", "text": "..." }
    """
    user_prompt = (
        f"Keyword: {keyword}\n\n"
        f"Nearby heading context (may include numbers/titles):\n{heading_lines_nearby}\n\n"
        f"Local text block where the keyword appears:\n{context_block}\n\n"
        f"Anchor hints (heuristic): {json.dumps(heading_hint, ensure_ascii=False)}\n\n"
        "Extract the best 'section' (title with its top-level number), the most specific 'sub section' number, "
        "and a short faithful excerpt that includes the keyword."
        "Do not give any repetative information."
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        content = resp.content.strip()
        # Try to parse JSON object
        obj = json.loads(extract_json_block(content))
        if isinstance(obj, dict) and {"section", "sub section", "text"} <= set(obj.keys()):
            return {
                "section": obj.get("section", "").strip(),
                "sub section": obj.get("sub section", "").strip(),
                "text": obj.get("text", "").strip(),
            }
    except Exception:
        return None
    return None


def extract_json_block(txt: str) -> str:
    """
    Extract first {...} JSON object from a text response.
    """
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        return txt[start:end+1]
    return txt


def nearby_heading_lines(page: PageInfo, center_line_idx: int, radius: int = 25) -> str:
    """
    Provide a wider band of lines around the hit to help the LLM see heading numbers above it.
    """
    lo = max(0, center_line_idx - radius)
    hi = min(len(page.lines), center_line_idx + radius + 1)
    block = "\n".join(page.lines[lo:hi]).strip()
    return block


# ---------- Pipeline ----------

def run_keyword_extraction(
    pdf_path: str,
    keywords: List[str],
    llm: AzureChatOpenAI,
    max_hits_per_keyword: Optional[int] = None,
    doc_client: Optional["DocumentIntelligenceClient"] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Orchestrates the whole flow and returns the final dict keyed by keyword.
    """
    # Load pages using Document Intelligence when available. Passing doc_client
    # allows callers to reuse an already-initialized client.
    # Identify candidate pages containing the keywords using a fast scan. Include
    # the previous page for additional heading context.
    candidate = find_candidate_pages(pdf_path, keywords)
    pages_to_analyze: Optional[List[int]] = None
    if candidate:
        pages_set = set(candidate)
        for i in candidate:
            if i > 0:
                pages_set.add(i - 1)
        pages_to_analyze = sorted(pages_set)
    # Load pages, restricting to candidate pages when available
    pages = load_pdf_pages(pdf_path, doc_client=doc_client, page_indices=pages_to_analyze)
    # Build a mapping from original 0-based page indices to their position in the
    # `pages` list. Because the pages list may be a sparse subset of the original
    # document (due to scanned-image pages being skipped or only candidate pages
    # being loaded), we need this mapping to correctly locate pages for hits.
    page_index_map: Dict[int, int] = {p.index0: idx for idx, p in enumerate(pages)}

    heading_anchor = build_heading_stack(pages)
    raw_hits = find_keyword_hits(pages, keywords)

    results: Dict[str, List[Dict[str, str]]] = {kw: [] for kw in keywords}

    for kw, hits in raw_hits.items():
        if not hits:
            continue

        counter = 0
        for hit in hits:
            original_idx = hit["page_idx0"]
            # Skip hits whose page is not loaded (e.g., scanned images or pages outside of analyzed subset)
            if original_idx not in page_index_map:
                continue
            pidx = page_index_map[original_idx]
            li = hit["line_idx"]
            page = pages[pidx]

            # Heuristic hint
            hint = heading_anchor.get((pidx, li), {"section": "", "sub_section": ""})

            # Heading evidence for the LLM (larger radius by default)
            head_evidence = nearby_heading_lines(page, li)

            # Ask LLM
            llm_obj = ask_llm_for_hit(
                llm=llm,
                keyword=kw,
                context_block=hit["context"],
                heading_hint=hint,
                heading_lines_nearby=head_evidence
            )

            # Fallback if LLM failed
            if not llm_obj:
                llm_obj = heuristic_fallback(hit, page, hint, kw, heading_text=head_evidence)

            # Assemble final record (with page number)
            record = {
                "section": llm_obj.get("section", "").strip(),
                "sub section": llm_obj.get("sub section", "").strip(),
                "page": page_span_for_block(page, hit["context"]),
                "text": llm_obj.get("text", "").strip(),
            }

            # Basic guard: ensure keyword (incl. plural variants) appears in the final text
            pat = _as_pattern(kw)
            if not pat.search(record["text"]):
                record["text"] = select_sentence_with_keyword(hit["context"], pat, kw_fallback=kw)

            # Deduplicate near-identical entries
            if not is_duplicate(results[kw], record):
                results[kw].append(record)

            counter += 1
            if max_hits_per_keyword and counter >= max_hits_per_keyword:
                break

        # Sort by sub section if numeric, else by page
        results[kw].sort(key=lambda r: (sort_key_for_subsection(r["sub section"]), safe_int(r["page"])))

    return results


def heuristic_fallback(hit: dict, page: PageInfo, hint: dict, keyword: str, heading_text: Optional[str] = None) -> dict:
    """
    Fallback method for determining the section, sub-section, and excerpt when
    the LLM is unavailable. It leverages hints from the heading stack, scans
    nearby lines for headings, and extracts a sensible snippet around the
    keyword. If a wider heading context is provided (via ``heading_text``), it
    will attempt to extract the most relevant section/subsection from that
    context.

    Parameters
    ----------
    hit : dict
        Information about the keyword hit (includes 'context' and line indices).
    page : PageInfo
        The page on which the hit occurs.
    hint : dict
        The section/sub-section hint derived from the heading stack.
    keyword : str
        The keyword being matched.
    heading_text : Optional[str]
        A string of lines around the hit that potentially includes headings.

    Returns
    -------
    dict
        A dictionary with keys 'section', 'sub section', and 'text'.
    """
    section = hint.get("section", "").strip()
    subsec = hint.get("sub_section", "").strip()

    # If provided, attempt to extract headings from the wider context to override
    # the hint. This helps when the heading stack missed a heading.
    if heading_text:
        heading_lines = heading_text.splitlines()
        context_headings = detect_headings(heading_lines)
        if context_headings:
            # Choose the last heading (closest to the hit)
            _, sec_num, title = context_headings[-1]
            sec_num = sec_num.strip()
            title = title.strip()
            if "." in sec_num:
                # Subsection
                subsec = sec_num
                if not section:
                    top_num = sec_num.split(".")[0]
                    section = f"{top_num} {title}".strip()
            else:
                section = f"{sec_num} {title}".strip()
                subsec = ""

    # Prefer the deepest inline numeric pattern in the hit context
    inline_nums = INLINE_REF_RE.findall(hit["context"])
    if inline_nums:
        best_inline = sorted(inline_nums, key=lambda x: x.count("."), reverse=True)[0]
        if not subsec or best_inline.count(".") > subsec.count("."):
            subsec = best_inline

    # Build a keyword pattern for selecting the excerpt
    try:
        pat = compile_keyword_pattern(keyword)
    except NameError:
        pat = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)

    text = select_sentence_with_keyword(hit["context"], pat, kw_fallback=keyword)
    return {"section": section, "sub section": subsec, "text": text}


def select_sentence_with_keyword(block: str, pattern_or_kw, kw_fallback: str = "") -> str:
    """
    Given a block of text and a keyword (or compiled pattern), return a short
    excerpt containing the keyword. The excerpt may include the sentence
    containing the keyword, the preceding non-numeric sentence, and possibly
    the following sentence for additional context. Numeric-only fragments
    (e.g., bullet numbers) are skipped.

    Parameters
    ----------
    block : str
        The context block surrounding a keyword occurrence.
    pattern_or_kw : Union[str, Pattern]
        The keyword or compiled regex pattern used to match the keyword.
    kw_fallback : str, optional
        A fallback plain string keyword to search if the pattern does not
        match any sentence.

    Returns
    -------
    str
        A concise excerpt containing the keyword and surrounding context.
    """
    pat = _as_pattern(pattern_or_kw)
    # Split block into sentences while keeping punctuation; remove empty fragments
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block.strip()) if s.strip()]

    def is_numeric_fragment(text: str) -> bool:
        """Return True if text is a numeric/alphabetic list marker or heading-like fragment.

        This covers roman numerals, alphabetical bullets, simple numbers, and
        section headings like "23 Cyber Security" or "23.1.4 Requirements".
        """
        # Roman numerals or simple alpha bullets (e.g., "i.", "(a)")
        if re.match(r"^[ivxlcdm]+\.?$", text.lower()):
            return True
        if re.match(r"^\(?[a-zA-Z]\)?\.?$", text.strip()):
            return True
        if re.match(r"^\d+[.)]?\s*$", text.strip()):
            return True
        # Section heading-like: starts with digits (with optional dot hierarchy) followed by uppercase/lowercase words
        if re.match(r"^\d+(?:\.\d+)*\s+\S+", text.strip()):
            return True
        return False

    # Helper to build excerpt from a target index
    def build_excerpt(index: int) -> str:
        prev = None
        # Find previous non-numeric sentence
        for j in range(index - 1, -1, -1):
            if not is_numeric_fragment(sents[j]):
                prev = sents[j]
                break
        # Find next non-numeric sentence
        nxt = None
        for j in range(index + 1, len(sents)):
            if not is_numeric_fragment(sents[j]):
                nxt = sents[j]
                break
        parts = []
        if prev:
            parts.append(prev)
        parts.append(sents[index])
        # Append next if the current excerpt is short
        if nxt and len(" ".join(parts)) < 80:
            parts.append(nxt)
        return " ".join(parts).strip()

    # First attempt: match using compiled pattern
    for i, s in enumerate(sents):
        if pat.search(s) and not is_numeric_fragment(s):
            return build_excerpt(i)
    # Fallback: match using raw keyword string
    if kw_fallback:
        for i, s in enumerate(sents):
            if re.search(re.escape(kw_fallback), s, re.IGNORECASE) and not is_numeric_fragment(s):
                return build_excerpt(i)
    # Last resort: return the first two non-empty sentences
    if not sents:
        return block.strip()[:600]
    if len(sents) == 1:
        return sents[0][:600]
    return f"{sents[0]} {sents[1]}"[:600]


def is_duplicate(existing: List[Dict[str, str]], cand: Dict[str, str]) -> bool:
    """
    Detect duplicates by comparing page number and a portion of the text **after**
    skipping the first few characters. This helps avoid false positives when
    headings or list markers differ but the core content is identical. Only
    consider entries duplicates when they are on the same page and the text
    beyond the skipped prefix matches.

    Parameters
    ----------
    existing : List[Dict[str, str]]
        List of existing result dictionaries for a given keyword.
    cand : Dict[str, str]
        The candidate result dictionary to check.

    Returns
    -------
    bool
        True if a duplicate is detected, False otherwise.
    """
    # Number of characters to skip at the beginning of the text when comparing.
    SKIP_CHARS = 40
    # Length of the substring used for comparison after skipping
    COMPARE_LENGTH = 100
    page = cand.get("page", "").strip()
    text = (cand.get("text", "") or "").strip()
    # Extract the comparison slice after skipping the prefix
    slice_cand = text[SKIP_CHARS:SKIP_CHARS + COMPARE_LENGTH].lower() if len(text) > SKIP_CHARS else text.lower()
    for r in existing:
        # Only consider duplicates on the same page
        page2 = r.get("page", "").strip()
        if page != page2:
            continue
        text2 = (r.get("text", "") or "").strip()
        slice_exist = text2[SKIP_CHARS:SKIP_CHARS + COMPARE_LENGTH].lower() if len(text2) > SKIP_CHARS else text2.lower()
        if slice_exist and slice_cand and slice_exist == slice_cand:
            return True
    return False


def sort_key_for_subsection(subsec: str) -> Tuple[int, ...]:
    """
    Convert '23.1.4' -> (23,1,4), '20' -> (20,), '' -> (999999,)
    """
    subsec = subsec.strip()
    if not subsec:
        return (999999,)
    parts = subsec.split(".")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except:
            out.append(99999)
    return tuple(out)


def safe_int(s: str) -> int:
    try:
        return int(str(s).split("-")[0].strip())
    except:
        return 999999

# ---------- Quick Page Scanning ----------

def find_candidate_pages(pdf_path: str, keywords: List[str]) -> List[int]:
    """
    Quickly scan the PDF using PyMuPDF to locate pages that contain any of the
    keywords. This avoids running the expensive Document Intelligence service
    on pages that do not contain keywords. Multi-word keywords are also
    matched across line breaks because ``compile_keyword_pattern`` uses
    ``\s+`` between words, which matches newlines.

    Parameters
    ----------
    pdf_path : str
        Local path to the PDF document.
    keywords : List[str]
        List of keywords to search for.

    Returns
    -------
    List[int]
        Sorted list of 0-based page indices that contain at least one keyword.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []
    patterns = {kw: compile_keyword_pattern(kw) for kw in keywords}
    candidate_pages = set()
    for i in range(len(doc)):
        page = doc[i]
        raw = page.get_text("text")
        norm = normalize_text(raw)
        if not any(ch.isalnum() for ch in norm):
            # Skip scanned/blank pages
            continue
        for kw, pat in patterns.items():
            if pat.search(norm):
                candidate_pages.add(i)
                break
    doc.close()
    return sorted(candidate_pages)


# ---------- Main ----------

def main():
    load_dotenv()
    # === Configure here (no argparse) ===
    PDF_PATH = r"C:\Users\INNIMK\OneDrive - ABB\Documents\ABB Projects\Cyber Security Bidding (CSB) Tool - DOC pack for ABB INdia\Roopashree_RFQ\DEP 32.01.20.12.pdf"   # <-- change me
    KEYWORDS = ["IEC 62443", "Firewall", "Backup", "Network Management", "System Hardening"]

    # Your provided AzureChatOpenAI setup (expects env vars or a configured openai object)
    # Ensure you have:
    #   os.environ["OPENAI_API_KEY"] = "..."
    #   os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"  (example)
    #   os.environ["AZURE_OPENAI_API_BASE"] = "https://your-endpoint.openai.azure.com/"
    #   LLM_MODEL = "gpt-4o"  (or your deployed name)
    openai.api_type = os.environ.get("AZURE_OPENAI_TYPE")
    openai.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
    openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    openai.api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    LLM_MODEL = os.environ.get("LLM_MODEL")

    llm = AzureChatOpenAI(
        azure_deployment=LLM_MODEL,
        openai_api_key=openai.api_key,
        openai_api_version=openai.api_version,
        azure_endpoint=openai.api_base,
    )

    # Initialize Document Intelligence client if credentials are available. This
    # client will be used to extract lines from the PDF instead of PyMuPDF.
    doc_client = None
    if _HAS_DOC_INTELLIGENCE:
        docai_endpoint = os.environ.get("DOCAI_ENDPOINT")
        docai_key = os.environ.get("DOCAI_KEY") 
        if docai_endpoint and docai_key:
            try:
                doc_client = DocumentIntelligenceClient(endpoint=docai_endpoint, credential=AzureKeyCredential(docai_key))
            except Exception:
                doc_client = None

    data = run_keyword_extraction(
        pdf_path=PDF_PATH,
        keywords=KEYWORDS,
        llm=llm,
        max_hits_per_keyword=None,  # keep all occurrences
        doc_client=doc_client,
    )

    # Save to Downloads as requested pattern in prior asks; adjust if needed.
    out_path = os.path.join(os.path.expanduser("~"), "Downloads", "keyword_hits_18.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete. Results saved to: {out_path}")


# Allow importing and direct execution
if __name__ == "__main__":
    main()