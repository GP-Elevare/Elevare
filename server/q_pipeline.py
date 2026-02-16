import sys
import json
import os
from models.qg.T5 import generate_questions
from models.qg.Text_Extractor import extract_text_from_pptx

def process_pptx(pptx_path):
    text = extract_text_from_pptx(pptx_path)
    q = generate_questions(text, 10)
    return {
        "pptx_path": pptx_path,
        "status": "processed",
        "questions": q
    }

if __name__ == "__main__":
    pptx_path = sys.argv[1]

    result = process_pptx(pptx_path)

    # Save JSON file in same folder as pptx
    output_path = "questions.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(json.dumps({
        "message": "Saved successfully",
        "output_path": output_path,
        "result": result
    }))
