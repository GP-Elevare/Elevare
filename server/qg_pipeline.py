
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













############################## CODE on demo, old preprocessing + T5
# import sys
# import json
# import os
# sys.stdout.flush()
# # from models.qg.qg_model import generate_questions
# # from models.qg.Text_Extractor import extract_text_from_pptx
# from models.qg.bloom_pipeline import run_bloom_pipeline  # <-- IMPORTANT IMPORT
# from models.qg.T5 import generate_questions
# from models.qg.Text_Extractor_old import extract_text_from_pptx

# ############################ Combined

# from pathlib import Path

# # current file: server/models/qg/bloom_pipeline.py
# BASE_DIR = Path(__file__).resolve().parent

# # go up to "models/"
# MODELS_DIR = BASE_DIR.parent

# # go up again to "server/"
# PROJECT_ROOT = MODELS_DIR.parent

# # input stays inside models/qg
# QG_INPUT_PATH = BASE_DIR / "models"/ "qg"/ "QG_data.json"
# # QG_INPUT_PATH = BASE_DIR / "QG_data.json"

# # output goes OUTSIDE models (but inside server root)
# OUTPUT_PATH = BASE_DIR / "QA_pairs_bloom.json"


# def process_pptx(pptx_path):
#     text = extract_text_from_pptx(pptx_path)

#     slides = text.split("--- Slide ")
#     slides = [s for s in slides if s.strip()]

#     output = []

#     for i, slide in enumerate(slides):
#         lines = slide.strip().split("\n")
#         context = "\n".join(lines[1:]).strip()

#         if not context:
#             continue

#         qa_pairs = generate_questions(context, 10)

#         for qa in qa_pairs:
#             output.append({
#                 "question": qa["question"],
#                 "answer": qa["answer"],
#                 "context": context   # IMPORTANT FOR BLOOM
#             })

#     return output

# if __name__ == "__main__":

#     pptx_path = sys.argv[1]

#     result = process_pptx(pptx_path)

#     output_path = "QG_data.json"   # or QA_ready.json

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     # RUN BLOOM PIPELINE
#     print("Running Bloom pipeline...")
#     final_output = run_bloom_pipeline(output_path, OUTPUT_PATH)

#     # RETURN RESULT (Node.js friendly)
#     print(json.dumps({
#         "message": "Pipeline complete (QG to Bloom)",
#         "output_path": str(OUTPUT_PATH),
#         "total": len(final_output)
#     }))























# #################################### QG alone working
# def process_pptx(pptx_path):
#     text = extract_text_from_pptx(pptx_path)

#     slides = text.split("--- Slide ")
#     slides = [s for s in slides if s.strip()]

#     output = []

#     for i, slide in enumerate(slides):
#         lines = slide.strip().split("\n")
#         context = "\n".join(lines[1:]).strip()

#         if not context:
#             continue

#         qa_pairs = generate_questions(context, 10)

#         for qa in qa_pairs:
#             output.append({
#                 "question": qa["question"],
#                 "answer": qa["answer"],
#                 "context": context   # ⭐ IMPORTANT FOR BLOOM
#             })

#     return output

# if __name__ == "__main__":
#     pptx_path = sys.argv[1]

#     result = process_pptx(pptx_path)

#     # Save JSON file in same folder as pptx
#     output_path = "questions.json"

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=4, ensure_ascii=False)

#     print(json.dumps({
#         "message": "Saved successfully",
#         "output_path": output_path,
#         "result": result
#     }))




# ############################## Bloom alone working
# from pathlib import Path

# # current file: server/models/qg/bloom_pipeline.py
# BASE_DIR = Path(__file__).resolve().parent

# # go up to "models/"
# MODELS_DIR = BASE_DIR.parent

# # go up again to "server/"
# PROJECT_ROOT = MODELS_DIR.parent

# # input stays inside models/qg
# QG_INPUT_PATH = BASE_DIR / "models"/ "qg"/ "QG_data.json"
# # QG_INPUT_PATH = BASE_DIR / "QG_data.json"

# # output goes OUTSIDE models (but inside server root)
# OUTPUT_PATH = BASE_DIR / "QA_pairs_bloom.json"


# def load_qg_file(path):
#     """Loads QG JSON output (no PPTX anymore)."""
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def flatten_qg_output(qg_data):
#     """
#     Converts QG format → Bloom pipeline format:
#     [
#       {question, reference_answer, student_answer}
#     ]
#     """
#     output = []

#     # Case 1: list format
#     if isinstance(qg_data, list):
#         for item in qg_data:
#             output.append({
#                 "question": item.get("question", ""),
#                 "reference_answer": item.get("reference_answer", ""),
#                 "student_answer": ""
#             })

#     # Case 2: dict format (story-based)
#     elif isinstance(qg_data, dict):
#         for story_id, story in qg_data.items():
#             qa_pairs = story.get("qa_pairs", [])
#             for qa in qa_pairs:
#                 output.append({
#                     "question": qa.get("question", ""),
#                     "reference_answer": qa.get("answer", ""),
#                     "student_answer": ""
#                 })

#     else:
#         raise ValueError(f"Unsupported format: {type(qg_data)}")

#     return output


# if __name__ == "__main__":

#     # RUN BLOOM PIPELINE
#     print("Running Bloom pipeline...")
#     final_output = run_bloom_pipeline(QG_INPUT_PATH, OUTPUT_PATH)

#     # RETURN RESULT (Node.js friendly)
#     print(json.dumps({
#         "message": "Pipeline complete (QG to Bloom)",
#         "output_path": str(OUTPUT_PATH),
#         "total": len(final_output)
#     }))


############# Newest QG from malak alone
# def process_pptx(pptx_path, max_questions=10):
#     """Extract text from PPTX slides and generate Q&A pairs using SG-CQG model."""
#     full_text = extract_text_from_pptx(pptx_path)
#     results = []

#     # Split by slide sections
#     slides = full_text.split("--- Slide ")
#     slides = [s for s in slides if s.strip()]  # remove empty

#     for slide_block in slides:
#         lines = slide_block.strip().split("\n")
#         slide_num = lines[0].replace("---", "").strip()
#         text = "\n".join(lines[1:]).strip()

#         if not text:
#             continue

#         print(f"Generating questions for slide {slide_num}...")
#         print(f"Slide text length: {len(text)}, content: {text[:100]}")
#         qa = generate_questions(text, max_questions)

#         results.append({
#             "slide_number": slide_num,
#             "num_questions": qa['num_questions'],
#             "questions": qa['questions'],
#             "answers": qa['answers']
#         })

#     return results



# if __name__ == "__main__":

#     print(f"Generating questio...")
#     pptx_path = sys.argv[1]
#     max_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 10
#     results = process_pptx(pptx_path, max_questions)

#     # add bloom code

#     # Flatten all slides' questions into the required format
#     output = []
#     for slide in results:
#         questions = slide.get("questions", [])
#         answers = slide.get("answers", [])
#         for q, a in zip(questions, answers):
#             output.append({
#                 "question": q,
#                 "reference_answer": a,
#                 "student_answer": ""
#             })

#     output_path = "QA_pairs.json"
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)

#     print(json.dumps({
#         "message": "Saved successfully",
#         "output_path": output_path,
#         "result": output
#     }))

