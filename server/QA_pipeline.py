import sys
import json
import os
from models.qa.qa_module import QAScorer

# Initialize the QA Model
qa_model = QAScorer()

# ══════════════════════════════════════════════════════════════════════════════

def process_qa(qa_json_path):
    """
    Load a QA pairs JSON file, grade every pair, and return the results.

    Args:
        qa_json_path: path to a JSON file containing a list of dicts with
                      keys: question, reference_answer, student_answer.

    Returns:
        list of graded result dicts.
    """
    print(f"\n[QA Pipeline] process_qa() called with: {qa_json_path}")

    if not qa_json_path or not os.path.exists(qa_json_path):
        print(f"[QA Pipeline] WARNING - QA file not found: {qa_json_path}. Returning []")
        return []

    with open(qa_json_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"[QA Pipeline] Loaded {len(qa_pairs)} QA pair(s) from {qa_json_path}")

    qa_results = qa_model.predict(qa_pairs)

    print(f"[QA Pipeline] process_qa() done — {len(qa_results)} result(s)")
    return qa_results


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # For a standalone file, we adjust the arguments.
    # Expected usage: python process_qa_only.py <input_qa_json> <output_json>
    
    qa_json_path = sys.argv[1] if len(sys.argv) > 1 else "qa_pairs_mock.json"
    output_file  = sys.argv[2] if len(sys.argv) > 2 else "qa_results.json"

    # The same safeguard you used before
    if not os.path.exists(qa_json_path): 
        print(f"[QA Pipeline] Fallback triggered. Using default mock file.")
        qa_json_path = "qa_pairs_mock.json"

    # Run the processing
    qa_results = process_qa(qa_json_path)

    # Save output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"qa_results": qa_results}, f, indent=2)

    print(f"\nResults saved successfully!")
    print(f"QA results       -->  {output_file}")