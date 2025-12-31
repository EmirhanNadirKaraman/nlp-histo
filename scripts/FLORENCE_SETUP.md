# Florence-2 Setup Guide for Mac

## Prerequisites

1. **Install Poppler** (required for PDF to image conversion):
```bash
brew install poppler
```

2. **Install Python dependencies**:
```bash
pip install torch torchvision transformers pillow pdf2image
```

For faster processing on Apple Silicon Macs, PyTorch will automatically use MPS (Metal Performance Shaders).

**Note**: The Florence-2-large-ft model will be downloaded automatically (~2.5GB) on first run.

## Usage

### Test Florence-2 Extraction Standalone

Extract from a single PDF:
```bash
python scripts/florence_extractor.py files/organized_pdfs/PMC10296831_dermatopathology-10-00026.pdf
```

Custom output location:
```bash
python scripts/florence_extractor.py files/organized_pdfs/PMC10296831_dermatopathology-10-00026.pdf \
  --output out/florence/custom_output.json
```

### Run 3-Way Comparison (PDFfigures2 + Docling + Florence-2)

```bash
python scripts/extract_and_visualize.py files/organized_pdfs/PMC10296831_dermatopathology-10-00026.pdf
```

This will:
- ✓ Skip PDFfigures2 (already exists)
- ✓ Skip Docling (already exists)
- ▶ Run Florence-2 extraction
- ▶ Create 3-way comparison visualization

Skip Florence-2 if needed:
```bash
python scripts/extract_and_visualize.py files/organized_pdfs/PMC10296831_dermatopathology-10-00026.pdf --skip-florence
```

## Output Files

For a PDF named `PMC10296831_dermatopathology-10-00026.pdf`:

- **Florence-2 JSON**: `out/florence/PMC10296831_dermatopathology-10-00026_florence.json`
- **Comparison PDF**: `out/comparisons/PMC10296831_dermatopathology-10-00026_comparison.pdf`

## Color Legend in Visualization

- **Blue solid**: PDFfigures2 figures
- **Green solid**: PDFfigures2 tables
- **Pink dashed**: Docling figures
- **Orange dashed**: Docling tables
- **Purple dashed**: Florence-2 figures
- **Bright orange dashed**: Florence-2 tables

## Performance Notes

- **Model**: Uses Florence-2-large-ft (fine-tuned) for best accuracy (~2.5GB)
- **First run**: Model will be downloaded automatically - this only happens once
- **Apple Silicon Macs**: Automatically uses MPS acceleration for faster processing
- **Intel Macs**: Uses CPU (slower but still works)
- **Processing time**: ~10-20 seconds per page depending on hardware
- **Multi-task detection**: Runs 3 different Florence-2 tasks per page and combines results for better coverage

## Troubleshooting

### "poppler not found"
```bash
brew install poppler
```

### "MPS backend not available"
This is normal on Intel Macs. The script will automatically fall back to CPU.

### Model download fails
Check your internet connection. The model is downloaded from Hugging Face and cached locally.

### Out of memory
For large PDFs, the script processes one page at a time to minimize memory usage. If you still have issues:
- Close other applications
- Reduce DPI in `florence_extractor.py` (line 104: change `dpi=300` to `dpi=150`)
