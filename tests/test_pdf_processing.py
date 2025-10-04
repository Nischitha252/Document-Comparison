"""
Unit tests for PDF processing utilities.

These tests verify that the helper functions for deriving synonyms
and chunking text behave as expected.  Run the tests with
``python -m unittest discover tests``.
"""

import unittest

try:
    # Import functions.  These imports may fail if optional
    # dependencies like langchain_core are not installed in the test
    # environment.  In that case we will skip the tests.
    from rfq_cybersecurity.utils.pdf_processing import derive_synonyms, chunk_section  # type: ignore
    FUNCTIONS_AVAILABLE = True
except Exception:
    derive_synonyms = None  # type: ignore
    chunk_section = None  # type: ignore
    FUNCTIONS_AVAILABLE = False


class TestPdfProcessing(unittest.TestCase):
    """Tests for functions in pdf_processing."""

    def test_derive_synonyms(self) -> None:
        """derive_synonyms should return lowercase terms and partials."""
        if not FUNCTIONS_AVAILABLE:
            self.skipTest("Optional dependencies are missing")
        heading = "Windows patches"
        synonyms = derive_synonyms(heading)  # type: ignore[misc]
        # The full heading in lowercase should be present
        self.assertIn("windows patches", synonyms)
        # The last word 'patches' should be extracted
        self.assertIn("patches", synonyms)

    def test_chunk_section(self) -> None:
        """chunk_section should produce overlapping chunks of the given size."""
        if not FUNCTIONS_AVAILABLE:
            self.skipTest("Optional dependencies are missing")
        text = "abcdefghijklmnopqrstuvwxyz"
        base_meta = {
            "section_num": "1",
            "section_title": "Test",
            "start_page": 1,
            "end_page": 1,
            "page_breaks": [],
        }
        # Create chunks of size 5 with an overlap of 2
        chunks = chunk_section(text, base_meta, chunk_size=5, chunk_overlap=2)  # type: ignore[misc]
        # Expect ceil((26-5)/(5-2)) + 1 chunks = 8
        self.assertEqual(len(chunks), 8)
        # Each chunk should contain the correct slice of text
        expected_first_chunk = text[:5]
        self.assertEqual(chunks[0].page_content, expected_first_chunk)
        # The second chunk should start at position 3 (5 - 2)
        expected_second_chunk = text[3:8]
        self.assertEqual(chunks[1].page_content, expected_second_chunk)


if __name__ == "__main__":
    unittest.main()