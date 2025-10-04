"""
Report generation service.

This module defines a high‑level function for processing a PDF
document and generating summarised content for each cybersecurity
heading.  The heavy lifting—loading the PDF, chunking text,
performing similarity search and querying the language model—is
performed here.  By isolating this logic from the Streamlit UI you
can test and iterate on it independently and reuse it in other
contexts (e.g. a batch processor or API endpoint).
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import tempfile
from typing import Any, Callable, Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from ..utils.pdf_processing import derive_synonyms, chunk_section
from ..utils.report_utils import json_to_docx  # noqa: F401 (used for type hinting)
from .llm_service import get_chat_model, get_embeddings

# Import the package logger
from ..logging_config import get_logger

logger = get_logger(__name__)


def generate_report_parallel(
    pdf_path: str,
    headings: List[str],
    progress_callback: Callable[[float], None] | None = None,
) -> Dict[str, Dict[str, str]]:
    """Process a PDF and generate summarised content for each heading in parallel.

    Parameters
    ----------
    pdf_path: str
        Path to the uploaded PDF file.
    headings: list[str]
        List of headings (cybersecurity topics) to search for.
    progress_callback: callable, optional
        A function that accepts a float in [0, 1] to report progress.

    Returns
    -------
    dict
        A mapping from each heading to a dict of section identifiers and
        summarised content.
    """
    # Emit an info log indicating the start of processing
    logger.info("Starting report generation for %d headings", len(headings))
    # Instantiate embeddings and language model clients
    try:
        embeddings = get_embeddings()
        llm = get_chat_model()
    except Exception as exc:
        logger.error("Failed to initialise OpenAI clients: %s", exc)
        raise
    # Load and parse the PDF into page objects using PyPDFLoader
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as exc:
        logger.error("Failed to load PDF '%s': %s", pdf_path, exc)
        raise
    # Parse the PDF into structured sections; replicate original logic
    sections: List[Dict[str, Any]] = []
    section_dict: Dict[str, str] = {}
    current_num: str | None = None
    current_title: str | None = None
    current_text: str = ""
    current_start_page: int | None = None
    current_end_page: int | None = None
    current_page_breaks: List[int] = []
    for page_doc in pages:
        page_text = page_doc.page_content
        page_num = page_doc.metadata.get("page", None)
        if page_num is None:
            page_num = len(sections) + 1
        for line in page_text.splitlines():
            if not line.strip():
                continue
            match = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", line.strip())
            if match:
                # Flush the previous section if present
                if current_num is not None:
                    content = current_text.strip()
                    if content:
                        sections.append(
                            {
                                "number": current_num,
                                "title": current_title,
                                "content": content,
                                "start_page": current_start_page,
                                "end_page": current_end_page,
                                "page_breaks": current_page_breaks.copy(),
                            }
                        )
                        section_dict[current_num] = current_title  # type: ignore
                    current_text = ""
                    current_page_breaks = []
                current_num = match.group(1)
                current_title = match.group(2).strip()
                current_start_page = page_num
                current_end_page = page_num
                continue
            if current_num is not None:
                prev_text = current_text.rstrip()
                if not current_text:
                    current_text = line
                else:
                    if prev_text.endswith("-"):
                        current_text = prev_text[:-1] + line.strip()
                    elif prev_text.endswith((".", ":", "?", "!")):
                        current_text += "\n" + line
                    else:
                        current_text += " " + line.strip()
                current_end_page = page_num
        if current_num is not None:
            current_page_breaks.append(len(current_text))
    # Append the last section if any
    if current_num is not None:
        content = current_text.strip()
        if content:
            sections.append(
                {
                    "number": current_num,
                    "title": current_title,
                    "content": content,
                    "start_page": current_start_page,
                    "end_page": current_end_page,
                    "page_breaks": current_page_breaks.copy(),
                }
            )
            section_dict[current_num] = current_title  # type: ignore
    # Chunk the sections
    chunk_docs: List[Document] = []
    for sec in sections:
        base_meta = {
            "section_num": sec["number"],
            "section_title": sec["title"],
            "start_page": sec["start_page"],
            "end_page": sec["end_page"],
            "page_breaks": sec["page_breaks"],
        }
        chunk_docs.extend(chunk_section(sec["content"], base_meta))
    # Log the number of sections and chunks created
    logger.debug("Parsed %d sections from PDF", len(sections))
    logger.debug("Generated %d chunks for similarity search", len(chunk_docs))
    # If no chunks were produced, fall back to page level
    if not chunk_docs:
        for idx, page in enumerate(pages):
            meta = {
                "section_num": str(page.metadata.get("page", idx + 1)),
                "section_title": f"Page {idx + 1}",
                "start_page": page.metadata.get("page", idx + 1),
                "end_page": page.metadata.get("page", idx + 1),
                "page_breaks": [],
                "chunk_offset": 0,
            }
            chunk_docs.append(Document(page_content=page.page_content, metadata=meta))
    # Build the vector store
    if chunk_docs:
        try:
            vectorstore = FAISS.from_documents(chunk_docs, embeddings)
        except Exception as exc:
            logger.error("Failed to build vector store: %s", exc)
            vectorstore = None
    else:
        vectorstore = None
    result: Dict[str, Dict[str, str]] = {}

    # Local function to process a single heading
    def _process_heading(heading: str) -> Dict[str, str]:
        heading_results: Dict[str, str] = {}
        # Derive search terms from the heading
        synonyms = derive_synonyms(heading)
        relevant_chunks: List[Document] = []
        # Attempt to use the vector store to narrow down relevant chunks
        if vectorstore is not None:
            search_hits: List[Document] = []
            k = 10
            for syn in synonyms:
                try:
                    hits = vectorstore.similarity_search(syn, k=k)
                    search_hits.extend(hits)
                except Exception:
                    continue
            seen_keys = set()
            for doc in search_hits:
                meta = doc.metadata
                unique_key = (meta.get("section_num"), meta.get("chunk_offset"))
                if unique_key in seen_keys:
                    continue
                if any(syn in doc.page_content.lower() for syn in synonyms):
                    seen_keys.add(unique_key)
                    relevant_chunks.append(doc)
        # If nothing was found via vector search, fall back to scanning all chunks
        if not relevant_chunks:
            for doc in chunk_docs:
                if any(syn in doc.page_content.lower() for syn in synonyms):
                    relevant_chunks.append(doc)
        # Log how many chunks will be summarised
        logger.debug("Heading '%s': %d relevant chunks", heading, len(relevant_chunks))
        # Deduplicate and summarise each relevant chunk
        local_seen_keys = set()
        for doc in relevant_chunks:
            meta = doc.metadata
            unique_key = (meta.get("section_num"), meta.get("chunk_offset"))
            if unique_key in local_seen_keys:
                continue
            local_seen_keys.add(unique_key)
            sec_num: str = meta["section_num"]  # type: ignore
            sec_title: str = meta["section_title"]  # type: ignore
            section_text: str = doc.page_content
            other_syns = [s for s in synonyms if s != heading.lower()]
            if other_syns:
                syn_list = ", ".join(other_syns)
                syn_str = f" or its synonyms ({syn_list})"
            else:
                syn_str = ""
            prompt = (
                f"Please extract all information related to \"{heading}\"{syn_str} from the "
                f"following section and combine it into a concise, self-contained statement.\n"
                "- Use only the information provided in the section; do not include any "
                "outside or generic knowledge.\n"
                "- Retain key terminology and phrasing from the original text where it enhances clarity.\n"
                f"- Include the exact term \"{heading}\" in your summary at least once.\n"
                "- Use clear, grammatically correct English.\n"
                "- Focus solely on this section; omit any references to other sections.\n\n"
                f"Section {sec_num} - {sec_title}:\n\"\"\"\n"
                f"{section_text}\n"
                "\"\"\""
            )
            try:
                llm_response = llm.invoke([HumanMessage(content=prompt)])
                if hasattr(llm_response, "content"):
                    snippet = llm_response.content.strip()
                elif isinstance(llm_response, list) and hasattr(llm_response[0], "content"):
                    snippet = llm_response[0].content.strip()  # type: ignore[index]
                else:
                    snippet = str(llm_response).strip()
            except Exception as e:
                snippet = f"[LLM call failed: {e}]"
            # Determine page references for the snippet
            positions: List[int] = []
            section_text_lower = section_text.lower()
            for syn in synonyms:
                positions += [m.start() for m in re.finditer(re.escape(syn), section_text_lower)]
            positions = sorted(set(positions))
            chunk_offset = meta.get("chunk_offset", 0)
            start_page = meta["start_page"]  # type: ignore
            page_breaks = meta.get("page_breaks", [])
            # Translate character index into page number
            def idx_to_page(idx: int) -> int:
                for i, boundary in enumerate(page_breaks):
                    if idx < boundary:
                        return start_page + i
                return start_page + len(page_breaks)
            if positions:
                abs_positions = [pos + chunk_offset for pos in positions]
                abs_positions.sort()
                first_idx, last_idx = abs_positions[0], abs_positions[-1]
                snippet_start_page = idx_to_page(first_idx)
                snippet_end_page = idx_to_page(last_idx)
            else:
                snippet_start_page = meta["start_page"]  # type: ignore
                snippet_end_page = meta["end_page"]  # type: ignore
            page_ref = (
                f"(Page {snippet_start_page})"
                if snippet_start_page == snippet_end_page
                else f"(Page {snippet_start_page}-{snippet_end_page})"
            )
            display_title = sec_title
            # Adjust display title for generic headings
            if display_title.lower() == "general":
                parent_num = sec_num.rsplit(".", 1)[0] if "." in sec_num else None
                if parent_num and parent_num in section_dict:
                    parent_title = section_dict[parent_num]
                    parent_title = (
                        parent_title.title() if parent_title and parent_title.isupper() else parent_title
                    )
                    display_title = f"{parent_title} - {display_title}"
            if display_title and display_title.isupper():
                display_title = display_title.title()
            key = f"Section {sec_num} ({display_title})"
            snippet_entry = f"{snippet} {page_ref}"
            heading_results[key] = snippet_entry
        return heading_results

    # Execute headings in parallel
    processed_count = 0
    total_headings = len(headings)
    if progress_callback:
        progress_callback(0.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_heading = {executor.submit(_process_heading, h): h for h in headings}
        for future in concurrent.futures.as_completed(future_to_heading):
            heading = future_to_heading[future]
            try:
                result[heading] = future.result()
            except Exception as e:
                result[heading] = {"error": f"Failed to process heading '{heading}': {e}"}
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count / total_headings)
    logger.info("Finished processing all headings")
    return result