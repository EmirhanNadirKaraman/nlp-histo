import sys
import json
from pathlib import Path
import fitz
import importlib.util
from collections import Counter
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import re

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

from collections import Counter as _Counter
from parsers.layout_utils import merge_rects, DOCLING_MASK_TYPES
from parsers.layout_utils import (
    SKIP_TYPES, fix_ligatures, build_table_bboxes, build_picture_pages,
    bbox_overlaps, centroid_inside,
)
from collections import defaultdict

from parsers.layout_utils import (
    FIG_NUM_RE, TAB_NUM_RE, nearest_caption, parse_caption_num, union_bbox, count_panels,
)

# Import database modules
from database import get_db_connection, Document, TextElement, Figure, Table

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

class CombinedPipeline: 
    def __init__(self): 
        self.PDF_PATH = Path('files/organized_pdfs/PMC1448691_his_2369.pdf')
        self.PMCID = 'PMC1448691'
        self.CAPTION_PATTERN = re.compile(r'^(Table|Figure)\s+\d+', re.IGNORECASE)

        self.DOCLING_OUTPUT_DIR = Path('out/docling_full')
        self.MASKED_PDF_DIR = Path('out/masked_pdfs')
        self.TEXT_OUTPUT_DIR = Path('out/text')
        self.TABLES_DIR = Path('files/tables')
        self.FIGURES_DIR = Path('files/figures')
        self.VISUALIZATION_DIR = Path('out/visualization')

        self.BASELINE = 'masked'    # change to any key in results to shift the reference

        self.load_modules()
        self.create_directories()
        self.converter = self.get_converter()
        self.load_tatr()

    def load_tatr(self): 
        print('🔄 Loading TATR model...')
        self.tatr_processor = AutoImageProcessor.from_pretrained('microsoft/table-transformer-detection')
        self.tatr_model = AutoModelForObjectDetection.from_pretrained('microsoft/table-transformer-detection')
        self.tatr_model.eval()

        print('✅ TATR model loaded')
        print(f"   Labels: {self.tatr_model.config.id2label}")

    def main(self):
        all_elements, doc = self.extract_layout_with_docling()

        elements_raw, reconstructed_elements = self.reconstruct_tables(
            all_elements=all_elements, 
            doc=doc
        )

        self.save_json_file_with_reconstruction(elements=reconstructed_elements, doc=doc)
        self.visualize_pdfs()
        masked_pdf_path, masked_elements = self.create_masked_pdf()
        text_elements = self.extract_layout_from_masked_pdf(path=masked_pdf_path, elements=reconstructed_elements)
        
        tatr_detections, page_rects, combined_text_els = self.table_detection_with_tatr(elements=reconstructed_elements)
        self.visualize_tatr_detections(tatr_detections)
        self.visualize_combined_detections(
            page_rects=page_rects, 
            tatr_detections=tatr_detections, 
            docling_elements=reconstructed_elements
        )

        results = self.run_all_combinations(
            elements_raw=elements_raw, 
            reconstructed_elements=reconstructed_elements, 
            text_elements=text_elements,
            combined_text_elements=combined_text_els
        )

        self.compare_results(results=results)

        self.save_results_and_diffs(
            results=results, 
            all_elements=reconstructed_elements
        )

        table_data, figure_data = self.crop_table_and_figures(masked_elements=masked_elements)
        self.ingest_to_database(
            results=results, 
            table_data=table_data, 
            figure_data=figure_data
        )

    def load_modules(self):
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        
        mask_tables = load_module('mask_tables', project_root / 'scripts/docling_files/mask_tables.py')
        visualize = load_module('visualize', project_root / 'scripts/visualize_docling_full.py')
        text_proc = load_module('text_proc', project_root / 'parsers/text_processing.py')

        self.process_pdf_with_masking = mask_tables.process_pdf_with_masking
        self.reconstruct_tables_from_lists = visualize.reconstruct_tables_from_lists
        self.visualize_pdf = visualize.visualize_full_layout
        self.ContextAwareStitcher = text_proc.ContextAwareStitcher
        self.remove_citations = text_proc.remove_citations

        print("✅ Imports successful!")


    def create_directories(self):
        for d in [
            self.DOCLING_OUTPUT_DIR, self.MASKED_PDF_DIR, self.TEXT_OUTPUT_DIR, 
            self.TABLES_DIR, self.FIGURES_DIR, self.VISUALIZATION_DIR
        ]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"📄 PDF: {self.PDF_PATH.name}")
        print(f"📁 PMCID: {self.PMCID}")


    def get_converter(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = False
        pipeline_options.do_ocr = True
        pipeline_options.images_scale = 2.0

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        return converter
    
    
    def extract_layout_with_docling(self): 
        print("Extracting layout from ORIGINAL PDF...")
        result = self.converter.convert(str(self.PDF_PATH))
        doc = result.document

        all_elements = []
        for element, level in doc.iterate_items():
            label = str(getattr(element, "label", "UNKNOWN")).split('.')[-1].upper()
            if not (hasattr(element, 'prov') and element.prov):
                continue
            prov = element.prov[0]
            bbox = prov.bbox
            text = ""
            if hasattr(element, 'text'):
                text = element.text
            elif hasattr(element, 'caption') and element.caption:
                text = element.caption.text
            all_elements.append({
                "type": label,
                "page": prov.page_no,
                "level": level,
                "bbox": {"x1": bbox.l, "y1": bbox.t, "x2": bbox.r, "y2": bbox.b},
                "text": text.strip() if text else None
            })

        # Reclassify TEXT elements that match caption patterns (e.g. "Table 6.", "Figure 3.")
        reclassified = 0
        for el in all_elements:
            if el.get('type') == 'TEXT' and self.CAPTION_PATTERN.match(el.get('text') or ''):
                el['type'] = 'CAPTION'
                reclassified += 1
        if reclassified:
            print(f"🔄 Reclassified {reclassified} TEXT elements as CAPTION")

        docling_json_path = self.DOCLING_OUTPUT_DIR / f"{self.PDF_PATH.stem}_full_layout.json"

        with open(docling_json_path, 'w') as f:
            json.dump({
                "metadata": {"pdf_path": str(self.PDF_PATH), "tool": "Docling", "extraction_date": datetime.now().isoformat()},
                "page_dimensions": {no: {"width": p.size.width, "height": p.size.height} for no, p in doc.pages.items()},
                "elements": all_elements
            }, f, indent=2)

        types = Counter([el['type'] for el in all_elements])
        print(f"\n✅ Original PDF: {len(all_elements)} elements")
        print(f"   Tables: {types.get('TABLE', 0)}")
        print(f"   Figures: {types.get('PICTURE', 0)}")
        print(f"   Captions: {types.get('CAPTION', 0)}")
        print(f"💾 {docling_json_path}")

        return all_elements, doc

    def reconstruct_tables(self, all_elements, doc):
        docling_json_path = self.DOCLING_OUTPUT_DIR / f"{self.PDF_PATH.stem}_full_layout.json"

        print("🔄 Reconstructing tables...")
        reconstructed_elements = self.reconstruct_tables_from_lists(str(docling_json_path))
        reconstructed_tables = [el for el in reconstructed_elements if el.get('type') == 'RECONSTRUCTED_TABLE']
        print(f"✅ Created {len(reconstructed_tables)} reconstructed tables")

        # Keep a snapshot before overwriting so the direct|no-reconstruct combo is available later.
        all_elements_raw = list(all_elements)

        # Replace all_elements with the merged result: sub-elements that form
        # a RECONSTRUCTED_TABLE are already removed from the list, so this avoids duplicates.
        n_before = len(all_elements)
        all_elements = reconstructed_elements
        print(f"📦 all_elements: {n_before} → {len(all_elements)} elements (absorbed {n_before - len(all_elements)} sub-elements into RECONSTRUCTED_TABLEs)")

        return all_elements_raw, all_elements

    def save_json_file_with_reconstruction(self, elements, doc):
        docling_reconstructed_json_path = self.DOCLING_OUTPUT_DIR / f"{self.PDF_PATH.stem}_full_reconstructed_layout.json"

        with open(docling_reconstructed_json_path, 'w') as f:
            json.dump({
                "metadata": {"pdf_path": str(self.PDF_PATH), "tool": "Docling", "extraction_date": datetime.now().isoformat()},
                "page_dimensions": {no: {"width": p.size.width, "height": p.size.height} for no, p in doc.pages.items()},
                "elements": elements
            }, f, indent=2)

        types = Counter([el['type'] for el in elements])
        print(f"\n✅ Original PDF: {len(elements)} elements")
        print(f"   Tables: {types.get('TABLE', 0)}")
        print(f"   Figures: {types.get('PICTURE', 0)}")
        print(f"   Captions: {types.get('CAPTION', 0)}")
        print(f"💾 {docling_reconstructed_json_path}")

    def visualize_pdfs(self):
        docling_json_path = self.DOCLING_OUTPUT_DIR / f"{self.PDF_PATH.stem}_full_layout.json"
        docling_reconstructed_json_path = self.DOCLING_OUTPUT_DIR / f"{self.PDF_PATH.stem}_full_reconstructed_layout.json"

        original_output_path = f"{self.PDF_PATH.stem}_full_layout.pdf"
        reconstructed_output_path = f"{self.PDF_PATH.stem}_full_reconstructed_layout.pdf"

        self.visualize_pdf(str(self.PDF_PATH), str(docling_json_path), output_path=original_output_path)
        self.visualize_pdf(str(self.PDF_PATH), str(docling_reconstructed_json_path), output_path=reconstructed_output_path)

    def create_masked_pdf(self):
        docling_json_path = self.DOCLING_OUTPUT_DIR / f"{self.PDF_PATH.stem}_full_layout.json"

        print("🔄 Masking tables and figures...")
        masked_pdf_path, _, masked_elements = self.process_pdf_with_masking(
            pdf_path=self.PDF_PATH,
            json_path=docling_json_path,
            output_dir=self.MASKED_PDF_DIR
        )
        print(f"\n✅ Masked PDF: {masked_pdf_path}")
        print(f"🚫 Masked: {len(masked_elements)} elements (tables/figures/captions)")

        return masked_pdf_path, masked_elements
    
    def extract_layout_from_masked_pdf(self, path, elements):
        print("Extracting layout from MASKED PDF...")
        result_masked = self.converter.convert(str(path))
        doc_masked = result_masked.document

        masked_pdf_elements = []
        for element, level in doc_masked.iterate_items():
            label = str(getattr(element, "label", "UNKNOWN")).split('.')[-1].upper()
            if not (hasattr(element, 'prov') and element.prov):
                continue
            prov = element.prov[0]
            bbox = prov.bbox
            text = ""
            if hasattr(element, 'text'):
                text = element.text
            elif hasattr(element, 'caption') and element.caption:
                text = element.caption.text
            masked_pdf_elements.append({
                "type": label,
                "page": prov.page_no,
                "level": level,
                "bbox": {"x1": bbox.l, "y1": bbox.t, "x2": bbox.r, "y2": bbox.b},
                "text": text.strip() if text else None
            })

        masked_json_path = self.DOCLING_OUTPUT_DIR / f"{path.stem}_full_reconstructed_layout.json"
        with open(masked_json_path, 'w') as f:
            json.dump({
                "metadata": {"pdf_path": str(path), "tool": "Docling", "extraction_date": datetime.now().isoformat()},
                "page_dimensions": {no: {"width": p.size.width, "height": p.size.height} for no, p in doc_masked.pages.items()},
                "elements": masked_pdf_elements
            }, f, indent=2)

        masked_types = Counter([el['type'] for el in masked_pdf_elements])
        print(f"\n✅ Masked PDF: {len(masked_pdf_elements)} elements")
        print("\n📊 COMPARISON:")
        print(f"   Original PDF: {len(elements)} elements")
        print(f"   Masked PDF:   {len(masked_pdf_elements)} elements")
        print(f"   Removed:      {len(elements) - len(masked_pdf_elements)} elements")
        print("\n🔍 Masked PDF element types:")
        for t, c in masked_types.most_common():
            print(f"   {t}: {c}")
        print(f"\n💾 {masked_json_path}")

        # Extract text elements from masked PDF for stitching
        text_element_types = {'TEXT', 'PARAGRAPH', 'SECTION_HEADER', 'TITLE', 'LIST', 'LIST_ITEM'}
        text_elements = [el for el in masked_pdf_elements if el.get('type') in text_element_types]
        print(f"\n📝 Text elements for processing: {len(text_elements)}")

        return text_elements

    def table_detection_with_tatr(self, elements):
        TATR_DPI = 150
        TATR_THRESHOLD = 0.95
        SCALE = 72 / TATR_DPI  # pixels -> PDF points

        doc_orig = fitz.open(str(self.PDF_PATH))
        tatr_detections = []  # list of {page, rect (fitz.Rect in PDF points), score, label}

        print(f'🔄 Running TATR on {len(doc_orig)} pages (DPI={TATR_DPI}, threshold={TATR_THRESHOLD})...')
        for page_num in range(len(doc_orig)):
            page = doc_orig[page_num]
            mat = fitz.Matrix(TATR_DPI / 72, TATR_DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

            inputs = self.tatr_processor(images=img, return_tensors='pt')
            with torch.no_grad():
                outputs = self.tatr_model(**inputs)
            tatr_results = self.tatr_processor.post_process_object_detection(
                outputs, threshold=TATR_THRESHOLD, target_sizes=[(img.height, img.width)]
            )[0]

            for score, label, box in zip(tatr_results['scores'], tatr_results['labels'], tatr_results['boxes']):
                x1_px, y1_px, x2_px, y2_px = box.tolist()
                # Pixel -> PDF points: both use top-left origin, no Y-flip needed
                tatr_detections.append({
                    'page': page_num + 1,
                    'rect': fitz.Rect(x1_px * SCALE, y1_px * SCALE, x2_px * SCALE, y2_px * SCALE),
                    'score': round(score.item(), 3),
                    'label': self.tatr_model.config.id2label[label.item()]
                })

        doc_orig.close()

        by_page = _Counter(d['page'] for d in tatr_detections)
        print(f'✅ TATR detected {len(tatr_detections)} tables:')
        for page_no in sorted(by_page):
            for d in [x for x in tatr_detections if x['page'] == page_no]:
                print(f'   Page {page_no}: {d["label"]} (score={d["score"]:.3f})')


        # ── Collect all mask rects per page ──────────────────────────────────────────

        # Build {page_no: [fitz.Rect, ...]} in PyMuPDF top-left coords
        page_rects = {}
        doc_tmp = fitz.open(str(self.PDF_PATH))

        for page_num in range(len(doc_tmp)):
            page_no = page_num + 1
            h = doc_tmp[page_num].rect.height
            rects = []

            # TATR detections (already in PDF-point top-left coords)
            for d in tatr_detections:
                if d['page'] == page_no:
                    rects.append(d['rect'])

            # Docling detections (bottom-left origin → flip Y)
            for el in elements:
                if el.get('type') in DOCLING_MASK_TYPES and el.get('page') == page_no:
                    b = el['bbox']
                    rects.append(fitz.Rect(b['x1'], h - b['y1'], b['x2'], h - b['y2']))

            if rects:
                page_rects[page_no] = merge_rects(rects)

        doc_tmp.close()

        total_before = sum(
            len([d for d in tatr_detections if d['page'] == p]) +
            len([el for el in elements if el.get('type') in DOCLING_MASK_TYPES and el.get('page') == p])
            for p in page_rects
        )
        total_after = sum(len(v) for v in page_rects.values())
        print(f'📦 Bounding-box merge: {total_before} raw rects → {total_after} merged rects')
        for p in sorted(page_rects):
            print(f'   Page {p}: {len(page_rects[p])} merged rect(s)')

        # ── Apply merged rects to build combined-masked PDF ───────────────────────────
        combined_masked_pdf_path = self.MASKED_PDF_DIR / f'{self.PDF_PATH.stem}_combined_masked.pdf'
        doc = fitz.open(str(self.PDF_PATH))
        total_redacted = 0

        for page_num in range(len(doc)):
            page_no = page_num + 1
            for rect in page_rects.get(page_no, []):
                page = doc[page_num]
                page.add_redact_annot(rect, fill=(1, 1, 1))
                total_redacted += 1
            doc[page_num].apply_redactions()

        doc.save(str(combined_masked_pdf_path))
        doc.close()
        print(f'\n✅ Combined-masked PDF: {combined_masked_pdf_path}')
        print(f'   Applied {total_redacted} merged redaction(s)')

        # ── Extract layout from combined-masked PDF with Docling ──────────────────────
        print('\n🔄 Extracting layout from combined-masked PDF...')
        result_combined = self.converter.convert(str(combined_masked_pdf_path))
        doc_combined = result_combined.document

        combined_masked_elements = []
        for element, level in doc_combined.iterate_items():
            label = str(getattr(element, 'label', 'UNKNOWN')).split('.')[-1].upper()
            if not (hasattr(element, 'prov') and element.prov):
                continue
            prov = element.prov[0]
            bbox = prov.bbox
            text = (getattr(element, 'text', '') or '').strip()
            combined_masked_elements.append({
                'type': label,
                'page': prov.page_no,
                'level': level,
                'bbox': {'x1': bbox.l, 'y1': bbox.t, 'x2': bbox.r, 'y2': bbox.b},
                'text': text or None
            })

        # ── Save combined layout JSON ─────────────────────────────────────────────────
        combined_json_path = self.DOCLING_OUTPUT_DIR / f'{combined_masked_pdf_path.stem}_layout.json'
        with open(combined_json_path, 'w') as f:
            json.dump({
                'metadata': {
                    'pdf_path': str(combined_masked_pdf_path),
                    'tool': 'Docling',
                    'source': 'combined_docling_tatr',
                    'tatr_threshold': TATR_THRESHOLD,
                    'extraction_date': datetime.now().isoformat()
                },
                'page_dimensions': {
                    no: {'width': p.size.width, 'height': p.size.height}
                    for no, p in doc_combined.pages.items()
                },
                'elements': combined_masked_elements
            }, f, indent=2)
        print(f'💾 Layout JSON: {combined_json_path}')

        # Store text elements for use in section 4 (extract_text is defined there)
        text_element_types = {'TEXT', 'PARAGRAPH', 'SECTION_HEADER', 'TITLE', 'LIST', 'LIST_ITEM'}
        combined_text_els = [el for el in combined_masked_elements if el.get('type') in text_element_types]

        combined_types = Counter([el['type'] for el in combined_masked_elements])
        print(f'\n✅ Combined-masked PDF: {len(combined_masked_elements)} elements')
        for t, c in combined_types.most_common():
            print(f'   {t}: {c}')
        print(f'\n📝 {len(combined_text_els)} text elements ready — run section 4 to extract and compare.')

        return tatr_detections, page_rects, combined_text_els
        
    def visualize_tatr_detections(self, tatr_detections):
        # ── Visualize TATR detections on the original PDF ────────────────────────────
        if not tatr_detections:
            print('⚠️  No TATR detections found — run the detection cell first.')
        else:
            tatr_vis_path = self.VISUALIZATION_DIR / f'{self.PDF_PATH.stem}_tatr_detections.pdf'

            # Color per label (TATR labels: 'table', 'table rotated')
            TATR_COLORS = {
                'table':         (1.0, 0.35, 0.0),   # Orange
                'table rotated': (0.8, 0.0,  0.8),   # Magenta
            }
            DEFAULT_TATR_COLOR = (1.0, 0.0, 0.0)  # Red fallback

            doc_vis = fitz.open(str(self.PDF_PATH))

            for d in tatr_detections:
                page = doc_vis[d['page'] - 1]
                color = TATR_COLORS.get(d['label'].lower(), DEFAULT_TATR_COLOR)
                rect  = d['rect']

                # Bounding box
                page.draw_rect(rect, color=color, width=2)

                # Label + score just above the top-left corner
                label_y = max(rect.y0 - 2, 8)  # keep inside page
                page.insert_text(
                    (rect.x0 + 2, label_y),
                    f"{d['label']} {d['score']:.2f}",
                    fontsize=7,
                    color=color,
                )

            # ── Legend on first page ─────────────────────────────────────────────────
            fp = doc_vis[0]
            lx, ly = 20, 20
            label_counts = Counter(d['label'] for d in tatr_detections)
            legend_h = 28 + len(label_counts) * 11
            fp.draw_rect(fitz.Rect(lx-5, ly-5, lx+175, ly+legend_h),
                        color=(0,0,0), fill=(1,1,1), width=0.5)
            fp.insert_text((lx, ly+10), 'TATR Detections:', fontsize=9, color=(0,0,0))
            for row, (lbl, cnt) in enumerate(sorted(label_counts.items())):
                y = ly + 22 + row * 11
                c = TATR_COLORS.get(lbl.lower(), DEFAULT_TATR_COLOR)
                fp.draw_line((lx, y), (lx+12, y), color=c, width=2)
                fp.insert_text((lx+16, y+3), f"{lbl} ({cnt})", fontsize=7, color=(0,0,0))

            doc_vis.save(str(tatr_vis_path))
            doc_vis.close()

            pages_hit = sorted(set(d['page'] for d in tatr_detections))
            print(f'✅ TATR visualization saved: {tatr_vis_path}')
            print(f'   {len(tatr_detections)} detection(s) on page(s): {pages_hit}')
            for d in tatr_detections:
                r = d['rect']
                print(f"   Page {d['page']:2d}: {d['label']:<16s} score={d['score']:.3f}  "
                    f"rect=({r.x0:.0f},{r.y0:.0f})-({r.x1:.0f},{r.y1:.0f}) pts")

    def visualize_combined_detections(self, page_rects, tatr_detections, docling_elements):
        # ── Visualize combined (Docling + TATR) merged detections on original PDF ─────
        if not page_rects:
            print('⚠️  No combined detections — run the masking cell first.')
        else:
            combined_vis_path = self.VISUALIZATION_DIR / f'{self.PDF_PATH.stem}_combined_detections.pdf'

            # Color scheme: TATR = orange, Docling = blue, merged boundary = green
            COLOR_TATR    = (1.0, 0.45, 0.0)   # Orange
            COLOR_DOCLING = (0.1, 0.45, 0.9)   # Blue
            COLOR_MERGED  = (0.0, 0.70, 0.2)   # Green

            doc_vis = fitz.open(str(self.PDF_PATH))

            for page_num in range(len(doc_vis)):
                page_no = page_num + 1
                page    = doc_vis[page_num]
                h       = page.rect.height

                # Draw individual TATR detections (orange, dashed-style with thin line)
                for d in tatr_detections:
                    if d['page'] == page_no:
                        page.draw_rect(d['rect'], color=COLOR_TATR, width=1.5)
                        page.insert_text(
                            (d['rect'].x0 + 2, max(d['rect'].y0 - 2, 8)),
                            f"TATR {d['score']:.2f}",
                            fontsize=6, color=COLOR_TATR
                        )

                # Draw individual Docling detections (blue)
                for el in docling_elements:
                    if el.get('type') in DOCLING_MASK_TYPES and el.get('page') == page_no:
                        b = el['bbox']
                        r = fitz.Rect(b['x1'], h - b['y1'], b['x2'], h - b['y2'])
                        page.draw_rect(r, color=COLOR_DOCLING, width=1.5)
                        page.insert_text(
                            (r.x0 + 2, max(r.y0 - 2, 8)),
                            el.get('type', ''),
                            fontsize=6, color=COLOR_DOCLING
                        )

                # Draw merged rects (green, thick) — what actually gets masked
                for rect in page_rects.get(page_no, []):
                    page.draw_rect(rect, color=COLOR_MERGED, width=2.5)

            # ── Legend on first page ─────────────────────────────────────────────────
            fp = doc_vis[0]
            lx, ly = 20, 20
            legend_items = [
                ('TATR detection',    COLOR_TATR,    1.5),
                ('Docling detection', COLOR_DOCLING, 1.5),
                ('Merged (masked)',   COLOR_MERGED,  2.5),
            ]
            legend_h = 28 + len(legend_items) * 12
            fp.draw_rect(fitz.Rect(lx-5, ly-5, lx+185, ly+legend_h),
                        color=(0,0,0), fill=(1,1,1), width=0.5)
            fp.insert_text((lx, ly+10), 'Combined Detections:', fontsize=9, color=(0,0,0))
            for row, (lbl, color, lw) in enumerate(legend_items):
                y = ly + 23 + row * 12
                fp.draw_line((lx, y), (lx+14, y), color=color, width=lw)
                fp.insert_text((lx+18, y+3), lbl, fontsize=7, color=(0,0,0))

            doc_vis.save(str(combined_vis_path))
            doc_vis.close()

            pages_hit = sorted(page_rects.keys())
            print(f'✅ Combined visualization saved: {combined_vis_path}')
            print(f'   Pages with detections: {pages_hit}')
            # print(f'   {total_before} raw rects → {total_after} merged rects')


    def extract_text(self, elements, table_bboxes=None, use_centroid=False):
        """Extract and stitch hierarchical text from a list of Docling elements.

        Args:
            elements:      List of element dicts (type, page, level, bbox, text).
            table_bboxes:  Optional {page: [bbox, ...]} – elements are skipped if
                        they overlap any of these bboxes.
            use_centroid:  If True, only skip an element when its centroid falls
                        inside a table/figure bbox (less aggressive than full overlap).

        Returns:
            (stitched_by_path, n_skipped)  where stitched_by_path is {path: [para, ...]}
        """
        overlap_fn = centroid_inside if use_centroid else bbox_overlaps
        picture_pages = build_picture_pages(elements)

        hierarchy = {}
        by_path   = defaultdict(list)
        skipped   = 0

        for el in elements:
            etype = el.get('type', '')
            level = el.get('level', 0)
            text  = fix_ligatures((el.get('text') or '').strip())
            if not text:
                continue

            if etype == 'SECTION_HEADER':
                hierarchy[level] = text
                hierarchy = {k: v for k, v in hierarchy.items() if k <= level}
            elif etype not in SKIP_TYPES:
                # Drop single-character tokens on pages with figures — these are
                # panel labels (a, b, c …) that Docling OCRs just outside the PICTURE bbox.
                if len(text) == 1 and el.get('page') in picture_pages:
                    skipped += 1
                    continue
                if table_bboxes:
                    page = el.get('page')
                    bbox = el.get('bbox')
                    if page and bbox and any(overlap_fn(bbox, tb) for tb in table_bboxes.get(page, [])):
                        skipped += 1
                        continue
                path_parts = [hierarchy[k] for k in sorted(hierarchy) if hierarchy.get(k)]
                by_path[' > '.join(path_parts) or 'Root'].append(text)

        stitcher = self.ContextAwareStitcher()
        stitched = {
            path: stitcher.reconstruct_paragraphs([self.remove_citations(t) for t in texts])
            for path, texts in by_path.items()
        }

        print("✅ extract_text() ready")
        return stitched, skipped

    def run_all_combinations(self, elements_raw, reconstructed_elements, text_elements, combined_text_elements): 
        COMBINATIONS = {
            'direct | raw': (
                elements_raw,
                build_table_bboxes(elements_raw, types=('TABLE', 'PICTURE'))
            ),
            'direct | reconstructed': (
                reconstructed_elements,
                build_table_bboxes(reconstructed_elements, types=('TABLE', 'RECONSTRUCTED_TABLE', 'PICTURE'))
            ),
            'masked': (text_elements, None),        # Docling-only masked PDF, regions already blank
            'combined | masked': (                  # Docling + TATR merged, regions already blank
                combined_text_elements, None
            ) if 'combined_text_els' in globals() else None,
        }
        # Drop any combinations that weren't set up (e.g. TATR cells not yet run)
        COMBINATIONS = {k: v for k, v in COMBINATIONS.items() if v is not None}

        results = {}  # name -> stitched_by_path
        print("Running text extraction combinations:")
        print(f"  {'Combination':<26s} {'Paths':>6s} {'Paras':>6s} {'Skipped':>8s}")
        print(f"  {'-'*50}")
        for name, (elements, bboxes) in COMBINATIONS.items():
            stitched, skipped = self.extract_text(elements, bboxes)
            results[name] = stitched
            n_paths = len(stitched)
            n_paras = sum(len(v) for v in stitched.values())
            print(f"  {name:<26s} {n_paths:>6d} {n_paras:>6d} {skipped:>8d}")

        return results

    def compare_results(self, results): 
        import difflib

        names = list(results.keys())
        others = [n for n in names if n != self.BASELINE]
        all_paths = sorted(set().union(*[set(results[n]) for n in names]))

        # ── Summary table ──────────────────────────────────────────────────────────────
        header = f"{'Path':<55s}" + "".join(f" {n[:10]:>10s}" for n in names)
        print(header)
        print("-" * len(header))
        for path in all_paths:
            counts = [len(results[n].get(path, [])) for n in names]
            marker = " *" if len(set(counts)) > 1 else ""
            print(f"{path[:53]:<55s}" + "".join(f" {c:>10d}" for c in counts) + marker)

        # ── Text diffs vs baseline ─────────────────────────────────────────────────────
        for other in others:
            print(f"\n{'='*80}")
            print(f"DIFF  {self.BASELINE!r}  vs  {other!r}")
            print('='*80)
            any_diff = False
            for path in all_paths:
                a = results[self.BASELINE].get(path, [])
                b = results[other].get(path, [])
                if a == b:
                    continue
                any_diff = True
                print(f"\n  [{path}]")
                for line in difflib.unified_diff(a, b, lineterm='', fromfile=self.BASELINE, tofile=other):
                    if line.startswith('+') and not line.startswith('+++'):
                        print(f"    + {line[1:][:120]}")
                    elif line.startswith('-') and not line.startswith('---'):
                        print(f"    - {line[1:][:120]}")
            if not any_diff:
                print("  (identical)")


    def save_results_and_diffs(self, results, all_elements): 
        def save_text(stitched, out_path, name, ordered=False):
            """Save stitched text to a .txt file. ordered=True preserves document section order."""
            with open(out_path, 'w') as f:
                order_label = 'document order' if ordered else 'alphabetical order'
                f.write(f"Document: {self.PMCID}  |  combination: {name}  |  {order_label}\n{'='*80}\n\n")
                items = stitched.items() if ordered else sorted(stitched.items())
                for path, paragraphs in items:
                    f.write(f"[{path}]\n{'-'*80}\n")
                    for p in paragraphs:
                        if p.strip():
                            f.write(f"{p}\n\n")
                    f.write("\n")

        # Save each combination — both alphabetical and document-order versions
        for name, stitched in results.items():
            slug = name.replace(' ', '_').replace('|', '-').replace('/', '-')

            # Alphabetical (existing behaviour)
            out_path = self.TEXT_OUTPUT_DIR / f"{self.PMCID}_{slug}.txt"
            save_text(stitched, out_path, name, ordered=False)
            print(f"💾 {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

            # Document order (new)
            out_path_ord = self.TEXT_OUTPUT_DIR / f"{self.PMCID}_{slug}_ordered.txt"
            save_text(stitched, out_path_ord, name, ordered=True)
            print(f"💾 {out_path_ord} ({out_path_ord.stat().st_size / 1024:.1f} KB)  [doc order]")

        # ── Centroid vs bbox comparison ──────────────────────────────────────────────────────────────────────────────
        _bboxes = build_table_bboxes(all_elements, types=('TABLE', 'RECONSTRUCTED_TABLE', 'PICTURE'))

        for label, use_centroid in [('bbox', False), ('centroid', True)]:
            stitched, skipped = self.extract_text(all_elements, _bboxes, use_centroid=use_centroid)
            for ordered, suffix in [(False, ''), (True, '_ordered')]:
                out_path = self.TEXT_OUTPUT_DIR / f"{self.PMCID}_direct_reconstructed_{label}{suffix}.txt"
                save_text(stitched, out_path, f"direct|reconstructed|{label}", ordered=ordered)
                order_tag = '  [doc order]' if ordered else ''
                print(f"💾 {out_path} ({out_path.stat().st_size / 1024:.1f} KB)  (skipped {skipped}){order_tag}")

    def crop_table_and_figures(self, masked_elements): 
        all_captions = [el for el in masked_elements if el.get('type') == 'CAPTION']

        # Use dicts keyed by ID so elements sharing the same caption are merged immediately
        merged_tables = {}
        merged_figures = {}

        for el in masked_elements:
            t = el.get('type')

            if t in ['TABLE', 'RECONSTRUCTED_TABLE']:
                raw_caption = el.get('caption') or ''
                if not raw_caption:
                    nearest = nearest_caption(el, all_captions)
                    raw_caption = nearest.get('text', '') if nearest else ''
                num = str(parse_caption_num(raw_caption, TAB_NUM_RE) or (len(merged_tables) + 1))
                if num not in merged_tables:
                    merged_tables[num] = {'table_id': num, 'caption': raw_caption or f'Table {num}',
                                        'page': el.get('page'), 'bbox': el.get('bbox'), 'type': t.lower()}
                else:
                    existing = merged_tables[num]
                    if existing['bbox'] and el.get('bbox'):
                        existing['bbox'] = union_bbox(existing['bbox'], el['bbox'])
                    if len(raw_caption) > len(existing['caption'] or ''):
                        existing['caption'] = raw_caption
                    if existing['type'] == 'table' and t.lower() == 'reconstructed_table':
                        existing['type'] = 'reconstructed_table'
                    print(f"   ⚠️  Merged duplicate Table {num}")

            elif t in ['FIGURE', 'PICTURE']:
                nearest = nearest_caption(el, all_captions)
                raw_caption = nearest.get('text', '') if nearest else ''
                num = str(parse_caption_num(raw_caption, FIG_NUM_RE) or (len(merged_figures) + 1))
                if num not in merged_figures:
                    merged_figures[num] = {'figure_id': num, 'caption': raw_caption or f'Figure {num}',
                                        'page': el.get('page'), 'bbox': el.get('bbox'), 'type': t.lower()}
                else:
                    existing = merged_figures[num]
                    if existing['bbox'] and el.get('bbox'):
                        existing['bbox'] = union_bbox(existing['bbox'], el['bbox'])
                    if len(raw_caption) > len(existing['caption'] or ''):
                        existing['caption'] = raw_caption
                    print(f"   ⚠️  Merged duplicate Figure {num} (same caption)")

        table_data = list(merged_tables.values())
        figure_data = list(merged_figures.values())

        doc = fitz.open(str(self.PDF_PATH))

        for t in table_data:
            if t['page'] and t['bbox']:
                p = doc[t['page'] - 1]
                h = p.rect.height
                b = t['bbox']
                r = fitz.Rect(b['x1'], h - max(b['y1'], b['y2']), b['x2'], h - min(b['y1'], b['y2']))
                pix = p.get_pixmap(clip=r, matrix=fitz.Matrix(2, 2))
                path = self.TABLES_DIR / f"{self.PMCID}_table_{t['table_id']}.png"
                pix.save(str(path))
                t['image_path'] = str(path)

        for f in figure_data:
            if not (f['page'] and f['bbox']):
                continue
            p = doc[f['page'] - 1]
            h = p.rect.height
            b = f['bbox']

            # Always save the full (merged) figure bbox as one image
            n_panels = count_panels(p, b, h)
            r = fitz.Rect(b['x1'], h - max(b['y1'], b['y2']), b['x2'], h - min(b['y1'], b['y2']))
            pix = p.get_pixmap(clip=r, matrix=fitz.Matrix(2, 2))
            path = self.FIGURES_DIR / f"{self.PMCID}_figure_{f['figure_id']}.png"
            pix.save(str(path))
            f['image_path'] = str(path)
            f['panels'] = n_panels

        doc.close()
        print(f"🖼️  Cropped {len(table_data)} tables, {len(figure_data)} figures")
        for td in table_data:
            print(f"   Table {td['table_id']}: {(td['caption'] or '')[:80]}")
        for fd in figure_data:
            panels = fd.get('panels', 1)
            panel_str = f" ({panels} panels)" if panels > 1 else ""
            print(f"   Figure {fd['figure_id']}{panel_str}: {(fd['caption'] or '')[:80]}")

        if table_data:
            with open(self.TABLES_DIR / f"{self.PMCID}_tables.json", 'w') as f:
                json.dump(table_data, f, indent=2)
        if figure_data:
            with open(self.FIGURES_DIR / f"{self.PMCID}_figures.json", 'w') as f:
                json.dump(figure_data, f, indent=2)
        print("💾 Metadata saved")

        return table_data, figure_data

    def ingest_to_database(self, results, table_data, figure_data):
        # Use the baseline combination for database ingestion
        stitched_by_path = results[self.BASELINE]

        # Prepare hierarchical elements for database
        # Need to convert stitched_by_path back into individual text elements with hierarchical info
        db_text_elements = []

        for path_string, stitched_paras in stitched_by_path.items():
            # Build path_list from path_string
            if path_string == 'Root':
                path_list = []
                depth = 0
            else:
                path_list = [part.strip() for part in path_string.split(' > ')]
                depth = len(path_list)
            
            # Each stitched paragraph becomes one text element
            for para in stitched_paras:
                if para.strip():
                    # Note: We don't have page info or references at this stage
                    # In production, you'd track these during the initial grouping
                    db_text_elements.append({
                        'path_list': path_list,
                        'path_string': path_string,
                        'depth': depth,
                        'text': para,
                        'references': {}  # Would be populated if we tracked them
                    })

        print(f"📦 Prepared {len(db_text_elements)} text elements for database")
        print(f"   Organized into {len(stitched_by_path)} hierarchical paths")

        # Preview first few elements
        print("\n🔍 Sample elements:")
        for i, elem in enumerate(db_text_elements[:3]):
            print(f"   {i+1}. Path: {elem['path_string']}")
            print(f"      Text: {elem['text'][:80]}...")
            print()

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        # Database ingestion
        db = get_db_connection()

        try:
            with db.session_scope() as session:
                # Check if document already exists
                existing = session.query(Document).filter_by(pmcid=self.PMCID).first()
                
                if existing:
                    print(f"⚠️  Document {self.PMCID} already exists in database")
                    force_reingest = False  # Set to True to delete and re-create
                    if force_reingest:
                        print("🗑️  Deleting existing document...")
                        session.delete(existing)
                        session.flush()
                    else:
                        print("   Skipping ingestion...")
                        raise Exception("Document exists - set force_reingest=True to overwrite")
                
                # Create document record
                doc = Document(
                    pmcid=self.PMCID,
                    filename=self.PDF_PATH.name,
                    file_path=str(self.PDF_PATH.absolute()),
                    title=f"Document {self.PMCID}",
                    journal=None,
                    publication_year=None,
                    text_source='pdf'
                )
                session.add(doc)
                session.flush()
                print(f"✅ Created document: {self.PMCID}")
                
                # Add text elements — skip duplicates on unique_path
                path_counts = defaultdict(int)
                inserted = skipped_dupes = 0
                for elem in db_text_elements:
                    path_string = elem['path_string']
                    position = path_counts[path_string]
                    path_counts[path_string] += 1
                    unique_path = f"{self.PMCID}/{path_string}/{position}" if path_string else f"{self.PMCID}/(Root)/{position}"
                    stmt = pg_insert(TextElement).values(
                        unique_path=unique_path,
                        document_id=doc.id,
                        path_list=elem['path_list'],
                        path_string=path_string,
                        depth=elem['depth'],
                        text_content=elem['text'],
                        position_in_section=position,
                        references=elem.get('references', {})
                    ).on_conflict_do_nothing(index_elements=['unique_path'])
                    result = session.execute(stmt)
                    if result.rowcount:
                        inserted += 1
                    else:
                        skipped_dupes += 1
                session.flush()
                print(f"✅ Added {inserted} text elements ({skipped_dupes} duplicates skipped)")
                
                # Add figures — skip duplicates on (document_id, figure_id)
                fig_inserted = fig_skipped = 0
                for fig in figure_data:
                    image_path = fig.get('image_path')
                    image_filename = Path(image_path).name if image_path else None
                    stmt = pg_insert(Figure).values(
                        document_id=doc.id,
                        figure_id=fig['figure_id'],
                        figure_label=f"Figure {fig['figure_id']}",
                        figure_number=fig['figure_id'],
                        caption_text=fig.get('caption'),
                        image_filename=image_filename,
                        image_path=image_path
                    ).on_conflict_do_nothing()
                    result = session.execute(stmt)
                    if result.rowcount:
                        fig_inserted += 1
                    else:
                        fig_skipped += 1
                session.flush()
                print(f"✅ Added {fig_inserted} figures ({fig_skipped} duplicates skipped)")
                
                # Add tables — skip duplicates on (document_id, table_id)
                tbl_inserted = tbl_skipped = 0
                for tbl in table_data:
                    image_path = tbl.get('image_path')
                    image_filename = Path(image_path).name if image_path else None
                    stmt = pg_insert(Table).values(
                        document_id=doc.id,
                        table_id=tbl['table_id'],
                        table_label=f"Table {tbl['table_id']}",
                        table_number=tbl['table_id'],
                        caption_text=tbl.get('caption'),
                        image_filename=image_filename,
                        image_path=image_path
                    ).on_conflict_do_nothing()
                    result = session.execute(stmt)
                    if result.rowcount:
                        tbl_inserted += 1
                    else:
                        tbl_skipped += 1
                session.flush()
                print(f"✅ Added {tbl_inserted} tables ({tbl_skipped} duplicates skipped)")
                
                print(f"\n🎉 Successfully ingested {self.PMCID} to database!")
                print(f"   Document ID: {doc.id}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main(): 
    combined_pipeline = CombinedPipeline()
    combined_pipeline.main()

if __name__ == '__main__': 
    main()
