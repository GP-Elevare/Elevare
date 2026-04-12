import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "valhalla/t5-base-qg-hl"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def _split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    # simple sentence split; works decently for slide text too
    parts = re.split(r"(?<=[.!?])\s+", text)
    # keep only meaningful chunks
    return [p.strip() for p in parts if len(p.strip()) >= 25]


# def generate_questions(text, max_questions=10):
#     """
#     Highlight-based question generation for valhalla/t5-base-qg-hl.

#     Returns:
#         dict: JSON-like dictionary containing generated questions.
#     """
#     sentences = _split_sentences(text)

#     questions = []
#     used = set()

#     # iterate more than max_questions to avoid weak/duplicate outputs
#     for sent in sentences[: max_questions * 4]:
#         prompt = f"generate question: <hl> {sent} <hl>"

#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=256
#         ).to(device)

#         out = model.generate(
#             **inputs,
#             max_length=64,
#             num_beams=6,
#             do_sample=False,              # deterministic reduces weird repetition
#             repetition_penalty=1.3,
#             no_repeat_ngram_size=3,
#             early_stopping=True
#         )

#         q = tokenizer.decode(out[0], skip_special_tokens=True).strip()

#         # basic filtering to avoid junky repeats
#         q_norm = q.lower()
#         if q and q_norm not in used and "value of x" not in q_norm:
#             used.add(q_norm)
#             questions.append(q)

#         if len(questions) >= max_questions:
#             break

#     return {
#         "num_questions": len(questions),
#         "questions": questions
#     }


def generate_questions(text, max_questions=10):
    sentences = _split_sentences(text)

    qa_pairs = []
    used = set()

    for sent in sentences[: max_questions * 4]:
        prompt = f"generate question: <hl> {sent} <hl>"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        out = model.generate(
            **inputs,
            max_length=64,
            num_beams=6,
            repetition_penalty=1.3
        )

        question = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        if not question:
            continue

        q_norm = question.lower()
        if q_norm in used:
            continue

        used.add(q_norm)

        qa_pairs.append({
            "question": question,
            "answer": sent   # ✅ THIS IS YOUR ANSWER (context sentence)
        })

        if len(qa_pairs) >= max_questions:
            break

    return qa_pairs