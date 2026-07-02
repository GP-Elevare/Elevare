################################# new prperocessing + T5
import sys
import json
import os
sys.stdout.flush()
# from models.qg.qg_model import generate_questions
from models.qg.bloom_pipeline import run_bloom_pipeline  # <-- IMPORTANT IMPORT
from models.qg.T5 import generate_questions

# --- NEW PREPROCESSING MODULE ---
from models.qg.Text_Extractor_new import SlideTextExtractor

# --- OLD EXTRACTOR (kept only as a fallback / reference, not used below) ---
# from models.qg.Text_Extractor_old import extract_text_from_pptx

############################ new preprocessing + T5

from pathlib import Path

# current file: server/models/qg/bloom_pipeline.py
BASE_DIR = Path(__file__).resolve().parent

# go up to "models/"
MODELS_DIR = BASE_DIR.parent

# go up again to "server/"
PROJECT_ROOT = MODELS_DIR.parent

# input stays inside models/qg
QG_INPUT_PATH = BASE_DIR / "models"/ "qg"/ "QG_data.json"
# QG_INPUT_PATH = BASE_DIR / "QG_data.json"

# output goes OUTSIDE models (but inside server root)
OUTPUT_PATH = BASE_DIR / "QA_pairs_bloom.json"


def process_pptx(pptx_path):
    # NEW: structured per-slide extraction instead of a "--- Slide N ---" marker string.
    # use_ocr=False by default (matches the decision from the standalone test phase).
    extractor = SlideTextExtractor(use_ocr=False)
    slides_data = extractor.extract_from_pptx(pptx_path)

    output = []

    for slide in slides_data:
        context = slide["story"].strip()

        if not context:
            continue

        qa_pairs = generate_questions(context, 10)

        for qa in qa_pairs:
            output.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "context": context   # IMPORTANT FOR BLOOM
            })

    return output

if __name__ == "__main__":

    pptx_path = sys.argv[1]

    result = process_pptx(pptx_path)

    output_path = "QG_data.json"   # or QA_ready.json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # RUN BLOOM PIPELINE
    print("Running Bloom pipeline...")
    final_output = run_bloom_pipeline(output_path, OUTPUT_PATH)

    # RETURN RESULT (Node.js friendly)
    print(json.dumps({
        "message": "Pipeline complete (QG to Bloom)",
        "output_path": str(OUTPUT_PATH),
        "total": len(final_output)
    }))










