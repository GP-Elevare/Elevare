
# import sys
# import json
# import os
# import subprocess
# sys.stdout.flush()
# from models.qg.bloom_pipeline import run_bloom_pipeline  # <-- UNCHANGED

# # --- NEW PREPROCESSING MODULE (from the previous integration step) ---
# from models.qg.Text_Extractor_new import SlideTextExtractor

# # --- OLD T5 MODEL (kept only as a fallback / reference, no longer used) ---
# # from models.qg.T5 import generate_questions

# # --- OLD EXTRACTOR (kept only as a fallback / reference, no longer used) ---
# # from models.qg.Text_Extractor_old import extract_text_from_pptx

# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent
# MODELS_DIR = BASE_DIR.parent
# PROJECT_ROOT = MODELS_DIR.parent

# QG_INPUT_PATH = BASE_DIR / "models" / "qg" / "QG_data.json"
# OUTPUT_PATH = BASE_DIR / "QA_pairs_bloom.json"

# # ── SG-CQG bridge config ──────────────────────────────────────────────────────
# # The SG-CQG model (semantic_graph.py + generate_conversation.py) needs
# # fairseq/allennlp/torch1.12, which lives in its own isolated conda env -
# # NOT the env this script (qg_pipeline.py) runs in.
# SGCQG_PYTHON = r"D:\conda_envs\sgcqg\python.exe"
# SGCQG_DIR = r"F:\content\SG-CQG model"          # where semantic_graph.py / generate_conversation.py / sgcqg_infer.py live
# SGCQG_SCRIPT = os.path.join(SGCQG_DIR, "sgcqg_infer.py")

# # Where the temp batch input/output for the subprocess call get written.
# SGCQG_TEMP_INPUT = BASE_DIR / "sgcqg_input.json"
# SGCQG_TEMP_OUTPUT = BASE_DIR / "sgcqg_output.json"


# def run_sgcqg(contexts):
#     """
#     contexts: list of {"id": ..., "context": ...}
#     Returns: list of {"id": ..., "context": ..., "qa_pairs": [{"question","answer"}, ...], "error"?: str}

#     Spawns ONE subprocess call for the whole batch (not one per slide) so the
#     heavy models (SRL, coref, 3x T5, RoBERTa classifier) only get loaded once.
#     """
#     with open(SGCQG_TEMP_INPUT, "w", encoding="utf-8") as f:
#         json.dump(contexts, f, indent=2, ensure_ascii=False)

#     print(f"Running SG-CQG model in its own env ({SGCQG_PYTHON})...")
#     print("This loads several large models on first call and can take a while, especially on CPU.")

#     result = subprocess.run(
#         [SGCQG_PYTHON, SGCQG_SCRIPT, str(SGCQG_TEMP_INPUT), str(SGCQG_TEMP_OUTPUT)],
#         cwd=SGCQG_DIR,          # so semantic_graph.py / generate_conversation.py resolve their own relative imports
#         capture_output=True,
#         text=True,
#     )

#     # Always show what the sgcqg subprocess printed - useful for debugging model
#     # loading / per-slide progress without digging through log files.
#     if result.stdout:
#         print(result.stdout)
#     if result.stderr:
#         print(result.stderr, file=sys.stderr)

#     if result.returncode != 0:
#         raise RuntimeError(f"SG-CQG subprocess failed with exit code {result.returncode}")

#     with open(SGCQG_TEMP_OUTPUT, "r", encoding="utf-8") as f:
#         return json.load(f)


# def process_pptx(pptx_path):
#     extractor = SlideTextExtractor(use_ocr=False)
#     slides_data = extractor.extract_from_pptx(pptx_path)

#     contexts = [
#         {"id": f"slide_{slide['slide_number']}", "context": slide["story"].strip()}
#         for slide in slides_data
#         if slide["story"].strip()
#     ]

#     if not contexts:
#         return []

#     sgcqg_results = run_sgcqg(contexts)

#     output = []
#     for item in sgcqg_results:
#         context = item["context"]
#         if item.get("error"):
#             print(f"WARNING: SG-CQG failed on '{item['id']}': {item['error']}", file=sys.stderr)
#         for qa in item.get("qa_pairs", []):
#             output.append({
#                 "question": qa["question"],
#                 "answer": qa["answer"],
#                 "context": context   # IMPORTANT FOR BLOOM
#             })

#     return output


# if __name__ == "__main__":

#     pptx_path = sys.argv[1]

#     result = process_pptx(pptx_path)

#     output_path = "QG_data.json"

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     # RUN BLOOM PIPELINE (unchanged)
#     print("Running Bloom pipeline...")
#     final_output = run_bloom_pipeline(output_path, OUTPUT_PATH)

#     # RETURN RESULT (Node.js friendly)
#     print(json.dumps({
#         "message": "Pipeline complete (QG to Bloom)",
#         "output_path": str(OUTPUT_PATH),
#         "total": len(final_output)
#     }))









