import os
import re
import sys

# ── Must be set BEFORE any imports that touch transformers/allennlp ───────────
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
# os.environ["TRANSFORMERS_CACHE"] = r"D:\content\models"
os.environ["HF_HOME"] = r"C:\Users\Nouran2026\.cache\huggingface"

print(f"Python version: {sys.version}")

# ── Path setup ────────────────────────────────────────────────────────────────
SGCQG_PATH = r"D:\content\SG-CQG model"
if SGCQG_PATH not in sys.path:
    sys.path.insert(0, SGCQG_PATH)

import importlib, types

LOCAL_MODEL_PATHS = {
    "coref": r"D:\content\models\coref_model",
    "spanbert": r"D:\content\models\spanbert-large-cased",
    "bert": r"D:\content\models\bert-base-uncased",
}


from allennlp.predictors.predictor import Predictor
from semantic_graph import Semantic_Graph_Constructor
from generate_conversation import Generate_Conversation_from_Graph


print("Loading SG-CQG models... (this may take a minute)")
sgc = Semantic_Graph_Constructor()
gen = Generate_Conversation_from_Graph()
print("Models loaded successfully.")


def generate_questions(text: str, max_questions: int = 10) -> dict:
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) < 15:
        return {"num_questions": 0, "questions": [], "answers": []}
    try:
        print(f"TEXT: {repr(text[:100])}")
        try:
            triples = sgc.build_linking_graph(text, max_questions)
        except Exception as e:
            print(f"Graph building failed: {e}")
            triples = []
        print(f"TRIPLES: {triples}")
        qa_pairs = gen.generate_full_conversation(text, triples, max_questions)
        print(f"RAW QA PAIRS: {qa_pairs}")
    except Exception as exc:
        print(f"[SG-CQG] Generation failed: {exc}")
        return {"num_questions": 0, "questions": [], "answers": []}
    
    questions, answers = [], []
    used = set()
    qa_pairs = [item for item in qa_pairs if item and item.strip()]

    for i in range(0, len(qa_pairs) - 1, 2):
        q = (qa_pairs[i]     or "").strip()
        a = (qa_pairs[i + 1] or "").strip()
        q_norm = q.lower()
        if not q or not a or q_norm in used:
            continue
        used.add(q_norm)
        questions.append(q)
        answers.append(a)
        if len(questions) >= max_questions:
            break
    return {
        "num_questions": len(questions),
        "questions":     questions,
        "answers":       answers,
    }