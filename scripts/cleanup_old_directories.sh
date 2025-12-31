#!/bin/bash
# Remove old subdirectories from files/masked_pdfs/
# Keeps only the PDF files

set -euo pipefail

MASKED_DIR="files/masked_pdfs"

cd "$(dirname "$0")/.."

echo "========================================================================"
echo "Cleaning up old directories in $MASKED_DIR/"
echo "========================================================================"

# Count directories
dir_count=$(find "$MASKED_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')

if [ "$dir_count" -eq 0 ]; then
    echo "✓ No directories to remove"
    exit 0
fi

echo "Found $dir_count directories to remove"
echo ""

# Remove all directories (keep files)
find "$MASKED_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

echo "✓ Removed $dir_count directories"
echo ""
echo "Remaining files:"
ls -lh "$MASKED_DIR" | grep "^-" | wc -l | xargs echo "  PDF files:"
echo "========================================================================"
