"""
test_preprocessing.py — standalone sanity check for Text_Extractor_new.py

Run this BEFORE wiring the new extractor into qg_pipeline.py, to confirm
it parses your real .pptx files correctly in your normal backend env.

Usage:
    python test_preprocessing.py path\\to\\deck.pptx
    python test_preprocessing.py path\\to\\deck.pptx --ocr
    python test_preprocessing.py path\\to\\deck.pptx --out .\\test_output --max-tokens 300

Prints a summary (slide count, non-empty slides, passage count/sizes) and
writes <stem>_coqa.json + <stem>_passages.json into --out (default: cwd).
"""

import argparse
import sys
from pathlib import Path

from Text_Extractor_new import process_pptx


def main():
    parser = argparse.ArgumentParser(description="Test the new PPTX preprocessing module.")
    parser.add_argument("pptx_path", help="Path to the .pptx file to process")
    parser.add_argument("--ocr", action="store_true",
                         help="Enable OCR on images (requires pytesseract + Tesseract binary)")
    parser.add_argument("--out", default=None,
                         help="Output directory for the JSON files (default: current dir)")
    parser.add_argument("--max-tokens", type=int, default=350,
                         help="Max words per CoQA passage (default: 350)")
    parser.add_argument("--quiet", action="store_true",
                         help="Suppress per-slide printout, show summary only")
    args = parser.parse_args()

    pptx_path = Path(args.pptx_path)
    if not pptx_path.exists():
        print(f"ERROR: file not found: {pptx_path}")
        sys.exit(1)

    result = process_pptx(
        str(pptx_path),
        use_ocr=args.ocr,
        save_json=True,
        output_dir=args.out,
        max_passage_tokens=args.max_tokens,
        verbose=not args.quiet,
    )

    non_empty = [s for s in result["slides_data"] if s["story"]]
    print("\n================= SUMMARY =================")
    print(f"File               : {pptx_path.name}")
    print(f"Total slides       : {len(result['slides_data'])}")
    print(f"Non-empty slides   : {len(non_empty)}")
    print(f"Big story length   : {len(result['big_story'].split())} words")
    print(f"Passages generated : {len(result['passages'])}")
    for i, p in enumerate(result["passages"], 1):
        print(f"  - passage {i}: {len(p.split())} words")
    print("=============================================")

    if not result["passages"]:
        print("\nWARNING: no passages were generated — check that the pptx "
              "actually has extractable text (not just images without alt-text).")
    else:
        print("\nOK: preprocessing ran successfully.")


if __name__ == "__main__":
    main()
