"""
Compare XML vs PDF Extraction Quality

This script compares the hierarchical extraction quality between XML and PDF
parsers for the same papers. Useful for validating PDF extraction accuracy.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.xml_parsers.hierarchical_parser import extract_hierarchy as extract_xml
from parsers.pdf_parsers.ensemble_parser import EnsemblePDFParser


class ExtractionComparator:
    """
    Compare XML and PDF extraction results for quality metrics.
    """

    def __init__(self):
        self.pdf_parser = EnsemblePDFParser()

    def compare_single_paper(
        self,
        xml_path: Path,
        pdf_path: Path
    ) -> Dict:
        """
        Compare XML and PDF extraction for a single paper.

        Args:
            xml_path: Path to XML file
            pdf_path: Path to PDF file

        Returns:
            Dictionary with comparison metrics
        """
        # Extract from both sources
        xml_data = extract_xml(str(xml_path))
        pdf_result = self.pdf_parser.extract_hierarchy(str(pdf_path))
        pdf_data = pdf_result['text_elements']

        # Calculate metrics
        metrics = {
            'pmcid': xml_path.stem,
            'xml_elements': len(xml_data),
            'pdf_elements': len(pdf_data),
            'pdf_method': pdf_result['extraction_method'],
            'pdf_success': pdf_result['success']
        }

        # Text completeness (character count ratio)
        xml_chars = sum(len(item['text']) for item in xml_data)
        pdf_chars = sum(len(item['text']) for item in pdf_data)
        metrics['xml_chars'] = xml_chars
        metrics['pdf_chars'] = pdf_chars
        metrics['text_coverage'] = pdf_chars / xml_chars if xml_chars > 0 else 0

        # Section detection accuracy
        xml_sections = set(item['path_string'] for item in xml_data)
        pdf_sections = set(item['path_string'] for item in pdf_data)

        common_sections = xml_sections & pdf_sections
        only_xml = xml_sections - pdf_sections
        only_pdf = pdf_sections - xml_sections

        metrics['xml_sections'] = len(xml_sections)
        metrics['pdf_sections'] = len(pdf_sections)
        metrics['common_sections'] = len(common_sections)
        metrics['only_xml_sections'] = len(only_xml)
        metrics['only_pdf_sections'] = len(only_pdf)

        # Precision and recall for section detection
        metrics['section_precision'] = (
            len(common_sections) / len(pdf_sections) if len(pdf_sections) > 0 else 0
        )
        metrics['section_recall'] = (
            len(common_sections) / len(xml_sections) if len(xml_sections) > 0 else 0
        )

        # F1 score
        if metrics['section_precision'] + metrics['section_recall'] > 0:
            metrics['section_f1'] = (
                2 * metrics['section_precision'] * metrics['section_recall'] /
                (metrics['section_precision'] + metrics['section_recall'])
            )
        else:
            metrics['section_f1'] = 0

        # Depth distribution
        xml_depths = Counter(item['depth'] for item in xml_data)
        pdf_depths = Counter(item['depth'] for item in pdf_data)

        metrics['xml_depth_dist'] = dict(xml_depths)
        metrics['pdf_depth_dist'] = dict(pdf_depths)

        # Store section details for inspection
        metrics['unique_to_xml'] = sorted(list(only_xml))[:10]  # First 10
        metrics['unique_to_pdf'] = sorted(list(only_pdf))[:10]  # First 10

        return metrics

    def print_comparison(self, metrics: Dict):
        """
        Print comparison metrics in a readable format.

        Args:
            metrics: Metrics dictionary from compare_single_paper
        """
        separator = "=" * 70

        print(f"\n{separator}")
        print(f"Comparison Results for {metrics['pmcid']}")
        print(separator)

        print(f"\nExtraction Method:")
        print(f"  PDF Parser: {metrics['pdf_method']}")
        print(f"  PDF Success: {metrics['pdf_success']}")

        print(f"\nElement Counts:")
        print(f"  XML elements: {metrics['xml_elements']}")
        print(f"  PDF elements: {metrics['pdf_elements']}")
        print(f"  Ratio (PDF/XML): {metrics['pdf_elements']/metrics['xml_elements']:.2f}")

        print(f"\nText Completeness:")
        print(f"  XML characters: {metrics['xml_chars']:,}")
        print(f"  PDF characters: {metrics['pdf_chars']:,}")
        print(f"  Coverage: {metrics['text_coverage']:.1%}")

        print(f"\nSection Detection:")
        print(f"  XML sections: {metrics['xml_sections']}")
        print(f"  PDF sections: {metrics['pdf_sections']}")
        print(f"  Common sections: {metrics['common_sections']}")
        print(f"  Precision: {metrics['section_precision']:.1%}")
        print(f"  Recall: {metrics['section_recall']:.1%}")
        print(f"  F1 Score: {metrics['section_f1']:.1%}")

        print(f"\nDepth Distribution:")
        print(f"  XML: {metrics['xml_depth_dist']}")
        print(f"  PDF: {metrics['pdf_depth_dist']}")

        if metrics['unique_to_xml']:
            print(f"\nSections only in XML (first 10):")
            for section in metrics['unique_to_xml']:
                print(f"  - {section}")

        if metrics['unique_to_pdf']:
            print(f"\nSections only in PDF (first 10):")
            for section in metrics['unique_to_pdf']:
                print(f"  - {section}")

        print()

    def compare_multiple_papers(
        self,
        xml_dir: Path,
        pdf_dir: Path,
        limit: int = None
    ) -> List[Dict]:
        """
        Compare extraction for multiple papers.

        Args:
            xml_dir: Directory with XML files
            pdf_dir: Directory with PDF files
            limit: Maximum number of papers to compare

        Returns:
            List of metrics dictionaries
        """
        # Find matching XML/PDF pairs
        xml_files = sorted(list(xml_dir.glob("*.nxml")))
        if not xml_files:
            xml_files = sorted(list(xml_dir.glob("*.xml")))

        if limit:
            xml_files = xml_files[:limit]

        all_metrics = []

        print(f"\nComparing {len(xml_files)} papers...")
        print()

        for i, xml_file in enumerate(xml_files, 1):
            pmcid = xml_file.stem
            pdf_file = None

            # Find corresponding PDF
            pdf_candidates = list(pdf_dir.glob(f"{pmcid}*.pdf"))
            if pdf_candidates:
                pdf_file = pdf_candidates[0]  # Take first match

            if not pdf_file:
                print(f"Skipping {pmcid}: No matching PDF found")
                continue

            print(f"[{i}/{len(xml_files)}] Comparing {pmcid}...")

            try:
                metrics = self.compare_single_paper(xml_file, pdf_file)
                all_metrics.append(metrics)

                # Print quick summary
                print(f"  Coverage: {metrics['text_coverage']:.1%}, "
                      f"Precision: {metrics['section_precision']:.1%}, "
                      f"Recall: {metrics['section_recall']:.1%}")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        return all_metrics

    def print_aggregate_statistics(self, all_metrics: List[Dict]):
        """
        Print aggregate statistics across all comparisons.

        Args:
            all_metrics: List of metrics dictionaries
        """
        if not all_metrics:
            print("\nNo metrics to aggregate")
            return

        separator = "=" * 70

        print(f"\n{separator}")
        print("Aggregate Statistics")
        print(separator)

        # Calculate averages
        avg_coverage = sum(m['text_coverage'] for m in all_metrics) / len(all_metrics)
        avg_precision = sum(m['section_precision'] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m['section_recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['section_f1'] for m in all_metrics) / len(all_metrics)

        print(f"\nPapers compared: {len(all_metrics)}")

        print(f"\nAverage Text Coverage: {avg_coverage:.1%}")
        print(f"  Range: {min(m['text_coverage'] for m in all_metrics):.1%} "
              f"- {max(m['text_coverage'] for m in all_metrics):.1%}")

        print(f"\nAverage Section Detection:")
        print(f"  Precision: {avg_precision:.1%}")
        print(f"  Recall: {avg_recall:.1%}")
        print(f"  F1 Score: {avg_f1:.1%}")

        # Parser usage
        marker_count = sum(1 for m in all_metrics if m['pdf_method'] == 'marker')
        nougat_count = sum(1 for m in all_metrics if m['pdf_method'] == 'nougat')

        print(f"\nParser Usage:")
        print(f"  Marker: {marker_count} ({marker_count/len(all_metrics):.1%})")
        print(f"  Nougat: {nougat_count} ({nougat_count/len(all_metrics):.1%})")

        # Success rate
        success_count = sum(1 for m in all_metrics if m['pdf_success'])
        print(f"\nSuccess Rate: {success_count}/{len(all_metrics)} ({success_count/len(all_metrics):.1%})")

        print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Compare XML vs PDF extraction quality'
    )
    parser.add_argument(
        '--xml-dir',
        type=str,
        default='files/organized_xmls',
        help='Directory with XML files'
    )
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='files/organized_pdfs',
        help='Directory with PDF files'
    )
    parser.add_argument(
        '--pmcid',
        type=str,
        help='Compare specific PMCID (e.g., PMC10047158)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of papers to compare'
    )

    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    pdf_dir = Path(args.pdf_dir)

    # Validate directories
    if not xml_dir.exists():
        print(f"Error: XML directory not found: {xml_dir}")
        return 1

    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return 1

    # Create comparator
    comparator = ExtractionComparator()

    # Check parser availability
    comparator.pdf_parser.print_availability()

    try:
        if args.pmcid:
            # Compare single paper
            xml_file = xml_dir / f"{args.pmcid}.nxml"
            if not xml_file.exists():
                xml_file = xml_dir / f"{args.pmcid}.xml"

            pdf_files = list(pdf_dir.glob(f"{args.pmcid}*.pdf"))

            if not xml_file.exists():
                print(f"Error: XML file not found for {args.pmcid}")
                return 1

            if not pdf_files:
                print(f"Error: PDF file not found for {args.pmcid}")
                return 1

            metrics = comparator.compare_single_paper(xml_file, pdf_files[0])
            comparator.print_comparison(metrics)

        else:
            # Compare multiple papers
            all_metrics = comparator.compare_multiple_papers(
                xml_dir,
                pdf_dir,
                limit=args.limit
            )

            comparator.print_aggregate_statistics(all_metrics)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
