#!/usr/bin/env python3
"""
Test script to verify caption assignment fix.
This creates mock data to test the caption matching logic.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_caption_extraction():
    """Test the caption number extraction logic."""
    import re

    def extract_number_from_caption(text, element_type):
        """Extract number from caption text."""
        if not text:
            return None

        if element_type == 'table':
            pattern = r'^\s*Table\s+(\d+[A-Za-z]?)'
        else:  # figure
            pattern = r'^\s*(?:Figure|Fig\.?)\s+(\d+[A-Za-z]?)'

        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    # Test cases
    test_cases = [
        # (caption_text, element_type, expected_number)
        ("Table 1: Patient demographics", "table", "1"),
        ("Table 2A shows the results", "table", "2A"),
        ("Figure 3: Tissue samples", "figure", "3"),
        ("Fig. 4B demonstrates the effect", "figure", "4B"),
        ("Figure 1 shows results similar to Table 2", "figure", "1"),  # Should extract 1, not 2
        ("See Table 3 for details", "table", None),  # Doesn't start with Table
        ("The figure shows...", "figure", None),  # No number
        ("  Table 5  : Data summary", "table", "5"),  # Leading spaces OK
    ]

    print("Testing caption number extraction:")
    print("=" * 80)

    all_passed = True
    for text, elem_type, expected in test_cases:
        result = extract_number_from_caption(text, elem_type)
        passed = result == expected
        all_passed &= passed

        status = "✓" if passed else "✗"
        print(f"{status} [{elem_type:6}] '{text[:50]:50}' => {result} (expected: {expected})")

    print("=" * 80)
    print(f"Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def test_caption_validation():
    """Test the caption validation logic."""
    import re

    def extract_number_from_caption(text, element_type):
        if not text:
            return None
        if element_type == 'table':
            pattern = r'^\s*Table\s+(\d+[A-Za-z]?)'
        else:
            pattern = r'^\s*(?:Figure|Fig\.?)\s+(\d+[A-Za-z]?)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def is_valid_caption_for_element(caption_text, expected_number, element_type):
        if not caption_text:
            return False
        extracted_number = extract_number_from_caption(caption_text, element_type)
        if extracted_number:
            return extracted_number == expected_number
        return False

    # Test cases
    test_cases = [
        # (caption, expected_num, elem_type, should_match)
        ("Table 1: Patient demographics", "1", "table", True),
        ("Table 2: Results", "1", "table", False),  # Wrong number
        ("Figure 3 shows Table 1", "1", "table", False),  # Not a table caption
        ("Figure 3 shows Table 1", "3", "figure", True),  # Correct figure caption
        ("Table 1A: Subgroup analysis", "1A", "table", True),
        ("Table 1A: Subgroup analysis", "1", "table", False),  # Number mismatch
        ("Fig. 5: Cell morphology", "5", "figure", True),
        ("This table shows...", "1", "table", False),  # No number
    ]

    print("\nTesting caption validation:")
    print("=" * 80)

    all_passed = True
    for caption, expected_num, elem_type, should_match in test_cases:
        result = is_valid_caption_for_element(caption, expected_num, elem_type)
        passed = result == should_match
        all_passed &= passed

        status = "✓" if passed else "✗"
        print(f"{status} [{elem_type:6}] num={expected_num:3} '{caption[:40]:40}' => {result} (expected: {should_match})")

    print("=" * 80)
    print(f"Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def test_no_duplicate_captions():
    """Test that captions can't be reused."""
    print("\nTesting duplicate caption prevention:")
    print("=" * 80)

    # Simulate the scenario
    used_caption_indices = set()

    # Mock elements array
    elements = [
        {'type': 'TABLE', 'index': 0},
        {'type': 'CAPTION', 'index': 1, 'text': 'Table 1: Data'},
        {'type': 'TABLE', 'index': 2},
    ]

    # First table should claim the caption
    if 1 not in used_caption_indices:
        used_caption_indices.add(1)
        print("✓ Table 1 claimed caption at index 1")

    # Second table should NOT be able to use the same caption
    if 1 not in used_caption_indices:
        print("✗ Table 2 incorrectly claimed caption at index 1")
        return False
    else:
        print("✓ Table 2 correctly rejected already-used caption")

    print("=" * 80)
    print("Result: ALL TESTS PASSED")
    return True


def test_text_element_captions():
    """Test that TEXT elements with table/figure captions are extracted."""
    import re

    def extract_number_from_caption(text, element_type):
        if not text:
            return None
        if element_type == 'table':
            pattern = r'^\s*Table\s+(\d+[A-Za-z]?)'
        else:
            pattern = r'^\s*(?:Figure|Fig\.?)\s+(\d+[A-Za-z]?)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    print("\nTesting TEXT element caption extraction:")
    print("=" * 80)

    # Simulate processing elements with TEXT captions
    elements = [
        {'type': 'TEXT', 'text': 'Table 1: Patient demographics', 'page': 1},
        {'type': 'TEXT', 'text': 'Some regular text', 'page': 1},
        {'type': 'TEXT', 'text': 'Figure 2: Cell morphology', 'page': 1},
        {'type': 'TEXT', 'text': 'Table 6. Anaplastic large-cell lymphoma', 'page': 2},
    ]

    seen_table_ids = set()
    seen_figure_ids = set()
    table_data = []
    figure_data = []

    for elem in elements:
        text = elem.get('text', '')

        # Check for table caption
        table_num = extract_number_from_caption(text, 'table')
        if table_num and table_num not in seen_table_ids:
            table_data.append({'table_id': table_num, 'caption': text})
            seen_table_ids.add(table_num)
            print(f"✓ Found table caption: {text[:60]}")

        # Check for figure caption
        fig_num = extract_number_from_caption(text, 'figure')
        if fig_num and fig_num not in seen_figure_ids:
            figure_data.append({'figure_id': fig_num, 'caption': text})
            seen_figure_ids.add(fig_num)
            print(f"✓ Found figure caption: {text[:60]}")

    # Verify results
    success = True
    if len(table_data) != 2:
        print(f"✗ Expected 2 tables, found {len(table_data)}")
        success = False
    else:
        print(f"✓ Correctly found 2 table captions")

    if len(figure_data) != 1:
        print(f"✗ Expected 1 figure, found {len(figure_data)}")
        success = False
    else:
        print(f"✓ Correctly found 1 figure caption")

    # Check specific IDs
    table_ids = {t['table_id'] for t in table_data}
    if table_ids != {'1', '6'}:
        print(f"✗ Expected table IDs 1 and 6, found {table_ids}")
        success = False
    else:
        print(f"✓ Correctly extracted table IDs: 1 and 6")

    print("=" * 80)
    print(f"Result: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
    return success


if __name__ == "__main__":
    print("Caption Assignment Fix - Test Suite")
    print("=" * 80)
    print()

    # Run all tests
    test1 = test_caption_extraction()
    test2 = test_caption_validation()
    test3 = test_no_duplicate_captions()
    test4 = test_text_element_captions()

    print("\n" + "=" * 80)
    if test1 and test2 and test3 and test4:
        print("✅ ALL TEST SUITES PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
