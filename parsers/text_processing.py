"""
Text Processing Utilities

Utilities for cleaning, stitching, and processing extracted text from PDFs.
Includes tools for handling split paragraphs, removing citations, and detecting
reference entries.
"""

import re
from typing import List, Union, Dict


class ContextAwareStitcher:
    """
    Stitches narrative text while preserving tables and figures.

    When a paragraph is interrupted by a table or figure, this class 'looks over'
    the interrupting object to find the continuation of the sentence and stitches
    them together intelligently.

    Example:
        >>> stitcher = ContextAwareStitcher()
        >>> paragraphs = [
        ...     "The results show that",  # Cut off mid-sentence
        ...     "Table 1: Results",       # Interrupting table
        ...     "the treatment was effective."  # Continuation
        ... ]
        >>> result = stitcher.reconstruct_paragraphs(paragraphs)
        >>> print(result)
        ["The results show that the treatment was effective.", "Table 1: Results"]
    """

    def reconstruct_paragraphs(self, paragraphs: List[Union[str, Dict]]) -> List[str]:
        """
        Reconstruct paragraphs by stitching split narratives.

        Args:
            paragraphs: List of paragraph strings or dicts with 'text' field

        Returns:
            List of reconstructed paragraphs with tables/figures preserved
        """
        if not paragraphs:
            return []

        # Convert to uniform format with type tags
        elements = []
        for para in paragraphs:
            if isinstance(para, dict):
                text = para.get('text', '')
            else:
                text = para

            # Classify paragraph type
            elem_type = self._classify_paragraph(text)

            elements.append({
                'type': elem_type,
                'text': text,
                'consumed': False
            })

        final_flow = []
        i = 0

        while i < len(elements):
            curr = elements[i]

            # If current block is a table or figure, preserve it
            if curr['type'] in ['table', 'figure']:
                final_flow.append(curr['text'])
                i += 1
                continue

            # If current is narrative, check if it's cut off
            if self._is_cut_off(curr['text']):
                # Look ahead for the next NARRATIVE block, skipping tables/figs
                next_text_idx = self._find_next_narrative(elements, i + 1)

                if next_text_idx is not None:
                    next_text_elem = elements[next_text_idx]

                    # Stitch them together
                    curr['text'] = self._merge_text(curr['text'], next_text_elem['text'])

                    # Mark the consumed element
                    elements[next_text_idx]['consumed'] = True

            # Add the (potentially stitched) block if not consumed
            if not curr.get('consumed'):
                final_flow.append(curr['text'])
            i += 1

        return final_flow

    def _classify_paragraph(self, text: str) -> str:
        """
        Classify paragraph as 'narrative', 'table', or 'figure'.

        Args:
            text: Text to classify

        Returns:
            One of: 'narrative', 'table', 'figure'
        """
        text_start = text.strip()[:50].lower()

        # Check if it's a table (starts with "Table" or is bullet points after a table)
        if text_start.startswith('table '):
            return 'table'

        # Check if it's a figure
        if text_start.startswith('figure '):
            return 'figure'

        # Check if it's table content (bullet points, pipes, etc.)
        if (text_start.startswith('-') or text_start.startswith('|') or
                text_start.startswith('*') or text_start.startswith('•')):
            # Likely table content, treat as table to skip over it
            return 'table'

        return 'narrative'

    def _is_cut_off(self, text: str) -> bool:
        """
        Check if text appears to be cut off mid-sentence.

        Args:
            text: Text to check

        Returns:
            True if text appears incomplete
        """
        t = text.strip()
        if not t:
            return False

        # Ends with hyphen (word break)
        # TODO: uncomment or remove this
        # if t.endswith('-'):
        #     return True

        # Ends with comma (sentence continues)
        if t.endswith(','):
            return True

        # Ends with lowercase letter or connector words
        last_word = t.split()[-1] if t.split() else ''
        connectors = ['and', 'or', 'but', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for']
        if last_word.lower() in connectors:
            return True

        # Ends with lowercase (likely mid-sentence)
        if t[-1].islower():
            return True

        return False

    def _find_next_narrative(self, elements: List[Dict], start_idx: int) -> Union[int, None]:
        """
        Find the next narrative block, skipping tables/figures.

        Args:
            elements: List of paragraph elements
            start_idx: Index to start searching from

        Returns:
            Index of next narrative, or None if not found
        """
        for j in range(start_idx, len(elements)):
            if elements[j]['type'] == 'narrative' and not elements[j].get('consumed'):
                return j
        return None

    def _merge_text(self, t1: str, t2: str) -> str:
        """
        Merge two text segments intelligently.

        Args:
            t1: First text segment
            t2: Second text segment

        Returns:
            Merged text
        """
        t1 = t1.rstrip()
        t2 = t2.lstrip()

        # Hyphen at end of t1: join without space to preserve compound words
        # (e.g. "ALK-" + "positive" → "ALK-positive").
        # True mid-word breaks (e.g. "lympho-" + "cyte") become "lympho-cyte",
        # which is preferable to "ALK- positive" with a spurious space.
        if t1.endswith('-'):
            return t1 + t2

        # Normal merge with space
        return t1 + " " + t2


def is_reference_entry(text: str) -> bool:
    """
    Check if a paragraph is a bibliography/reference entry.

    Reference entries typically:
    - Start with a number followed by period (e.g., "53. ")
    - Contain author names (Capital letter patterns)
    - Contain publication years (19xx or 20xx)
    - Contain journal/publication identifiers

    Args:
        text: Text to check

    Returns:
        True if this appears to be a reference entry

    Example:
        >>> is_reference_entry("1. Smith J, Jones A. Nature. 2020;123:456.")
        True
        >>> is_reference_entry("The results were significant.")
        False
    """
    text_stripped = text.strip()

    # Pattern 1: numbered reference  "53. Smith J, ..."
    if re.match(r'^\d+\.\s+', text_stripped):
        has_author  = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][\.,]', text_stripped))
        has_year    = bool(re.search(r'\b(19|20)\d{2}\b', text_stripped))
        has_journal = bool(re.search(r'\b(doi|DOI|http|www|PMID|ISBN)', text_stripped))
        if has_author or (has_year and has_journal) or (has_year and len(text_stripped) > 100):
            return True

    # Pattern 2: author-year style  "Lastname, F.; Lastname, F.; ..."
    # At least two authors separated by semicolons, each "Word, X." or "Word, X.Y."
    if len(re.findall(r'\w[\w\u00C0-\u024F]+,\s+[A-Z]\.', text_stripped)) >= 2:
        return True

    # Pattern 3: "Author FI, Author FI, et al. Title. Journal Year; ..."
    # Starts with an author-like token, "et al." appears in the first 150 chars, year present.
    if re.match(r'^[A-Z][a-z]+[\s,]+[A-Z]', text_stripped):
        if re.search(r'\bet\s+al\.', text_stripped[:150], re.IGNORECASE):
            if re.search(r'\b(19|20)\d{2}\b', text_stripped):
                return True

    return False


def remove_citations(text: str) -> str:
    """
    Remove citation numbers from text.

    Citations appear as numbers after periods or commas, like:
    - "text. 1 More text" → "text. More text"
    - "text. 19,20 More" → "text. More"
    - "text. 1-3 More" → "text. More"
    - "text. 4,5" → "text."

    Does NOT remove reference list numbers at start of lines like:
    - "14. Author name..." (preserved)

    Args:
        text: Text with citations

    Returns:
        Text with citations removed

    Example:
        >>> remove_citations("The study found. 12 These results")
        "The study found. These results"
        >>> remove_citations("14. Smith et al. 2020")
        "14. Smith et al. 2020"  # Start-of-line numbers preserved
    """
    # URLs (http/https)
    cleaned = re.sub(r'https?://\S+', '', text)

    # Bracket-style citations: [1], [1,2], [1-29], [3,11,21,22], [1–3]
    # Must contain only digits, commas, hyphens, or en-dashes — not e.g. [Table 1]
    cleaned = re.sub(r'\[\d+(?:[,–\-]\d+)*\]', '', cleaned)

    # After period: ". 1 ", ". 19,20 ", ". 4,5" (with or without trailing space/text)
    # Use negative lookbehind to avoid matching start of line
    cleaned = re.sub(r'(?<!\n)\.\s+\d+(?:[,–\-]\d+)*(?=\s|$)', '. ', cleaned)

    # After comma: ", 5 ", ", 1,2 "
    cleaned = re.sub(r',\s+\d+(?:[,–\-]\d+)*(?=\s|$)', ', ', cleaned)

    # Standalone citations in middle of text: " 1 ", " 19,20 "
    # (preceded by space and followed by space or end)
    cleaned = re.sub(r'(?<=\s)\d+(?:[,–\-]\d+)+(?=\s|$)', '', cleaned)

    # Clean up whitespace artifacts left by citation removal.
    # A space immediately before . , ; is never valid in English prose.
    cleaned = re.sub(r'\s+([.,;])', r'\1', cleaned)
    # Collapse any double spaces introduced by removals
    cleaned = re.sub(r'  +', ' ', cleaned)

    return cleaned


def clean_text(text: str, remove_refs: bool = True, strip_citations: bool = True) -> str:
    """
    Apply multiple text cleaning operations.

    Args:
        text: Text to clean
        remove_refs: If True, skip reference entries entirely (returns empty string)
        strip_citations: If True, remove citation numbers

    Returns:
        Cleaned text (may be empty if it's a reference entry)

    Example:
        >>> clean_text("The study found. 12 These results")
        "The study found. These results"
        >>> clean_text("1. Smith J. Nature. 2020", remove_refs=True)
        ""  # Reference entry removed
    """
    # Skip reference entries
    if remove_refs and is_reference_entry(text):
        return ""

    # Remove citations
    if strip_citations:
        text = remove_citations(text)

    return text
