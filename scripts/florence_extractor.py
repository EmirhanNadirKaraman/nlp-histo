import os
import re
import json
import torch
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path
# Use the specific Florence2 class instead of AutoModel
from transformers import AutoProcessor, Florence2ForConditionalGeneration

# --- CONFIGURATION ---
MODEL_ID = "florence-community/Florence-2-large" 
DPI = 150  # Balanced for M1 RAM
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class HistoPipeline:
    def __init__(self):
        print(f"🚀 Initializing HistoPipeline on {DEVICE.upper()}...")
        
        # 1. Load the processor
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        
        # 2. Use the specific Florence2 class to avoid ValueError
        # Using float16 for M1 acceleration
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32
        ).to(DEVICE).eval()

    def detect_objects(self, pil_img):
        """Uses Phrase Grounding for medical layout detection."""
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = "table. figure. micrograph. chart. diagram."
        prompt = task_prompt + text_input
        
        # Standardize image to RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
            
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(DEVICE)
        
        # Match model precision for MPS
        if DEVICE == "mps":
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=pil_img.size
        )
        return results[task_prompt]

    def convert_to_pdf_coords(self, bbox, img_size, page_rect):
        """Maps vision bounding boxes (pixels) to PyMuPDF (points)."""
        img_w, img_h = img_size
        pdf_w, pdf_h = page_rect.width, page_rect.height
        sx, sy = pdf_w / img_w, pdf_h / img_h
        
        return fitz.Rect(
            bbox[0] * sx, bbox[1] * sy, 
            bbox[2] * sx, bbox[3] * sy
        )

    def process_pdf(self, pdf_path, output_dir):
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return

        doc = fitz.open(str(pdf_path))
        print(f"📄 Processing: {pdf_path.name} ({len(doc)} pages)")
        
        try:
            images = convert_from_path(str(pdf_path), dpi=DPI)
        except Exception as e:
            print(f"❌ Rasterization failed: {e}")
            return

        final_data = {"filename": pdf_path.name, "pages": []}
        
        for i, (page, pil_img) in enumerate(zip(doc, images)):
            print(f"  [Page {i+1}] Running Vision Detection...")
            detections = self.detect_objects(pil_img)
            
            mask_rects = []
            objects_found = []
            for label, bbox in zip(detections['labels'], detections['bboxes']):
                pdf_rect = self.convert_to_pdf_coords(bbox, pil_img.size, page.rect)
                mask_rects.append(pdf_rect)
                objects_found.append({
                    "type": label, 
                    "bbox": [pdf_rect.x0, pdf_rect.y0, pdf_rect.x1, pdf_rect.y1]
                })

            # Spatial Masking Logic
            blocks = page.get_text("blocks")
            narrative_blocks = []
            for b in blocks:
                block_rect = fitz.Rect(b[:4])
                # Filter: Keep block if it doesn't overlap with a Table/Figure
                if not any(block_rect.intersects(m) for m in mask_rects):
                    clean_text = b[4].replace("\n", " ").strip()
                    if clean_text: narrative_blocks.append(clean_text)

            stitched_text = self.stitch_text(narrative_blocks)
            
            final_data["pages"].append({
                "page_num": i + 1,
                "narrative": stitched_text,
                "detected_objects": objects_found
            })

        os.makedirs(output_dir, exist_ok=True)
        out_json = Path(output_dir) / f"{pdf_path.stem}_clean.json"
        with open(out_json, "w", encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Exported to: {out_json}")

    def stitch_text(self, blocks):
        if not blocks: return ""
        text = blocks[0]
        for i in range(1, len(blocks)):
            if text.endswith("-"):
                text = text[:-1] + blocks[i]
            else:
                text += " " + blocks[i]
        return re.sub(r'\s+', ' ', text)

# --- RUN ---
if __name__ == "__main__":
    INPUT_FILE = "files/organized_pdfs/PMC1448691_his_2369.pdf"
    OUT_DIR = "out/florence"

    pipeline = HistoPipeline()
    pipeline.process_pdf(INPUT_FILE, OUT_DIR)