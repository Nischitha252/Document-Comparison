"""
PDF processing utilities.

Functions in this module implement the core text splitting logic
required for handling large PDF documents.  They are responsible for
deriving search terms from section headings and for breaking
sections into overlapping chunks so that the vector store can
represent them accurately.  These utilities are independent of any
external services and can be unit‑tested in isolation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.documents import Document

# Import the package logger
from ..logging_config import get_logger

logger = get_logger(__name__)


def derive_synonyms(heading: str) -> List[str]:
    """Derive a list of lowercase search terms from a heading.

    This function splits the heading on ``/`` to capture alternate
    terms and also extracts the last word of each part when it is
    sufficiently long (three characters or more and containing a
    letter).  This helps capture partial matches such as
    ``"patches"`` when the heading is ``"Windows patches"``.

    Parameters
    ----------
    heading: str
        The heading text from which to derive synonyms.

    Returns
    -------
    list[str]
        A list of lowercase synonyms and partial terms.
    """
    # Start with the lowercase version of the entire heading as the first synonym
    synonyms: List[str] = [heading.lower()]
    # Split the heading on '/' to handle alternate terms such as "foo/bar"
    parts = re.split(r"/", heading)
    for part in parts:
        # Remove surrounding whitespace from the part
        part = part.strip()
        if not part:
            # Skip empty parts
            continue
        lower_part = part.lower()
        # Add the lowercased part if it's new
        if lower_part not in synonyms:
            synonyms.append(lower_part)
        # Break the part into words to derive partial synonyms
        words = part.split()
        if len(words) > 1:
            # Consider only the last word for partial matching
            last_word = words[-1].strip("()[]{}")
            # Only consider words with three or more characters and at least one letter
            if len(last_word) >= 3 and re.search(r"[a-zA-Z]", last_word):
                lower_last = last_word.lower()
                if lower_last not in synonyms:
                    synonyms.append(lower_last)
    logger.debug("Derived synonyms for heading '%s': %s", heading, synonyms)
    return synonyms


def chunk_section(
    section_text: str,
    base_metadata: Dict[str, Any],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split a section's text into overlapping character chunks.

    Each chunk copies the base metadata and includes an additional
    ``chunk_offset`` indicating the starting character index of the
    chunk within the original section text.  Overlap is used so that
    context is preserved across chunk boundaries.

    Parameters
    ----------
    section_text: str
        The full text of the section to split.
    base_metadata: dict
        Metadata from the parent section (section number, title,
        pages, page breaks).
    chunk_size: int, optional
        The maximum number of characters per chunk (default is 2000).
    chunk_overlap: int, optional
        The number of characters of overlap between consecutive chunks
        (default is 200).

    Returns
    -------
    list[Document]
        A list of :class:`langchain_core.documents.Document`
        objects representing the chunks.
    """
    chunks: List[Document] = []
    text_len = len(section_text)
    # Start at the beginning of the section text
    offset = 0
    # Loop until we've consumed all characters
    while offset < text_len:
        # Determine the end index of the current chunk
        end = offset + chunk_size
        # Slice the chunk from the section text
        chunk_text = section_text[offset:end]
        # Copy the base metadata so each chunk gets its own dict
        metadata = base_metadata.copy()
        # Record where this chunk starts in the original section
        metadata["chunk_offset"] = offset
        # Append a new Document containing the chunk and its metadata
        chunks.append(Document(page_content=chunk_text, metadata=metadata))
        # Move the offset forward by (chunk_size - chunk_overlap) to introduce overlap
        if chunk_size > chunk_overlap:
            offset += chunk_size - chunk_overlap
        else:
            # Prevent an infinite loop if chunk_size <= chunk_overlap
            break
    logger.debug(
        "Chunked section into %d chunks (size=%d, overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks