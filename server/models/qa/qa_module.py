import os
import dspy
from dspy import InputField, OutputField, Signature
from typing import Literal


# ══════════════════════════════════════════════════════════════════════════════
#  DSPy / Groq setup
# ══════════════════════════════════════════════════════════════════════════════

print("[QAScorer] -- Module loading ------------------------------")
print(f"[QAScorer] GROQ_API_KEY found: {'YES' if os.environ.get('GROQ_API_KEY') else 'NO - will crash'}")

dspy.configure(
    lm=dspy.LM(
        model="groq/llama-3.3-70b-versatile",
        # This checks the env variable first, then falls back to your hardcoded key safely
        api_key=os.environ.get("GROQ_API_KEY", "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"),
        temperature=0.0,
        max_tokens=2048,
    )
)
print("[QAScorer] DSPy configured successfully")
print("[QAScorer] -- Module ready -------------------------------")


# ══════════════════════════════════════════════════════════════════════════════
#  Signature
# ══════════════════════════════════════════════════════════════════════════════

class AnswerScoringSignature(Signature):
    """
    You are a STRICT AI grader for an automated presentation training system.

    Grading rules:
    - STEP 0: Check if the student's answer addresses the question's specific subject.
      If the student answers a different question entirely, mark as incorrect immediately.
    - STEP 1: Determine the minimum threshold FROM THE QUESTION TEXT, not from the reference.
      The reference is a pool of possible correct answers — its length is irrelevant.
    - STEP 2: Count each distinct point SEPARATELY. A bullet listing 5 fields counts
      as 5 points, not 1.
    - STEP 3: For each individual point check: on-topic, factually correct, explicitly stated.
      All three must pass.
    - STEP 4: score = (valid_points / required_from_question) * 100, capped at 100.
      correct (>=80), partially correct (40-79), incorrect (<40).
    """

    question:         str = InputField(desc="The question the student was asked.")
    reference_answer: str = InputField(desc="The reference ground-truth answer.")
    student_answer:   str = InputField(desc="The student's answer to evaluate.")

    question_scope:    str       = OutputField(desc="Topic and minimum points required, extracted from question text.")
    alignment_check:   str       = OutputField(desc="Does the student's answer address the question? YES/NO + reason.")
    reasoning:         str       = OutputField(desc="Point-by-point evaluation with final score calculation.")
    key_points_missed: list[str] = OutputField(desc="Points the student missed to reach the minimum threshold.")
    hallucinations:    list[str] = OutputField(desc="Facts stated by the student that contradict the reference.")
    score:             int       = OutputField(desc="0-100 score.")
    label: Literal["correct", "partially correct", "incorrect"] = OutputField(
        desc="correct (>=80), partially correct (40-79), incorrect (<40)."
    )
    feedback: str = OutputField(desc="Constructive, coaching-style feedback for the student.")


# ══════════════════════════════════════════════════════════════════════════════
#  Public class
# ══════════════════════════════════════════════════════════════════════════════

class QAScorer:

    def __init__(self) -> None:
        print("[QAScorer] Initialising QAScorer ...")
        self._verifier = dspy.ChainOfThought(AnswerScoringSignature)
        print("[QAScorer] QAScorer ready")

    def predict(self, pairs: list[dict]) -> list[dict]:
        print(f"\n[QAScorer] == predict() called ==========================")
        print(f"[QAScorer] Pairs received : {len(pairs)}")

        if not pairs:
            print("[QAScorer] WARNING - empty pairs list, returning []")
            return []

        results: list[dict] = []

        for i, pair in enumerate(pairs):
            print(f"\n[QAScorer] -- Pair {i+1}/{len(pairs)} --------------------------")

            question  = pair.get("question", "")
            reference = pair.get("reference_answer", "")
            student   = pair.get("student_answer", "")

            print(f"[QAScorer]   Question  : {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"[QAScorer]   Reference : {reference[:80]}{'...' if len(reference) > 80 else ''}")
            print(f"[QAScorer]   Student   : {student[:80]}{'...' if len(student) > 80 else ''}")
            print(f"[QAScorer]   Calling LLM ...")

            try:
                pred = self._verifier(
                    question=question,
                    reference_answer=reference,
                    student_answer=student,
                )
                print(f"[QAScorer]   LLM responded successfully")
                print(f"[QAScorer]   Score     : {pred.score}")
                print(f"[QAScorer]   Label     : {pred.label}")
                print(f"[QAScorer]   Alignment : {pred.alignment_check[:80]}{'...' if len(pred.alignment_check) > 80 else ''}")
                print(f"[QAScorer]   Missed    : {pred.key_points_missed}")
                print(f"[QAScorer]   Halluc.   : {pred.hallucinations}")

                result = {
                    "question":          question,
                    "reference_answer":  reference,
                    "student_answer":    student,
                    "score":             pred.score,
                    "label":             pred.label,
                    "feedback":          pred.feedback,
                    "reasoning":         pred.reasoning,
                    "key_points_missed": pred.key_points_missed,
                    "hallucinations":    pred.hallucinations,
                    "question_scope":    pred.question_scope,
                    "alignment_check":   pred.alignment_check,
                }

            except Exception as exc:
                print(f"[QAScorer]   ERROR calling LLM: {exc}")
                result = {
                    "question":         question,
                    "reference_answer": reference,
                    "student_answer":   student,
                    "error":            str(exc),
                }

            results.append(result)
            print(f"[QAScorer]   Result appended ({len(results)} total so far)")

        print(f"\n[QAScorer] == predict() done - {len(results)} result(s) ==\n")
        return results