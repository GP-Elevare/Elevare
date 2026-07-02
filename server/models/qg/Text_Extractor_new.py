# -*- coding: utf-8 -*-
"""
Text_Extractor_new.py — PPTX -> CoQA-format context, for SG-CQG.

This is the preprocessing half of the SG-CQG pipeline, lifted out of the
Colab notebook and cleaned up to run in a normal backend environment
(no torch / fairseq / allennlp needed here — that's the SG-CQG model
itself, which stays in the separate python3.9 "sgcqg" env for the next
integration step).

Dependencies (install once in your normal backend venv):
    pip install python-pptx Pillow lxml nltk
    pip install pytesseract        # optional, only if you want OCR on images
                                    # also requires the Tesseract-OCR binary
                                    # installed separately on Windows:
                                    # https://github.com/UB-Mannheim/tesseract/wiki

First run only, download the nltk sentence tokenizer data:
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

NOTE (Windows): by default pip/nltk/huggingface caches land on C:\\Users\\<you>\\...
See the "avoid filling C:" notes given alongside this file for how to
redirect NLTK_DATA / pip cache / HF cache to another drive.

Fixes over the original extractor:
  1. URLs stripped from story text
  2. Lone slide-number strings ("2", "56") removed
  3. Title no longer duplicated in story
  4. Bullet points become individual sentences (not one collapsed line)
  5. Image alt-text extracted when content-relevant; decorative captions skipped
  6. Tables converted to natural-language sentences
  7. Speaker notes stored separately, then folded back in as extra sentences
  8. Repeated alt-text across slides deduplicated at the deck level
"""

import io
import re
import json
import logging
import unicodedata
from pathlib import Path
from typing import Optional

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

logger = logging.getLogger(__name__)

# ── Text cleaning ─────────────────────────────────────────────────────────────

_URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)
_SLIDE_NO = re.compile(r'^\d{1,3}$')
_ZERO_W = re.compile(r'[\u00a0\u200b\u200c\u200d\ufeff]')
_UNICODE_MAP = {
    '\u2019': "'", '\u2018': "'",
    '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-',
    '\u000b': ' ',   # vertical tab used by pptx for bullet breaks
}

# Some decks use logic symbols (from math/CS slides) that render oddly once
# stripped of formatting — spell them out instead of dropping them.
_LOGIC_MAP = {
    '\u2227': ' AND ',
    '\u2192': ' IMPLIES ',
    '\u00ac': ' NOT ',
    '\u2200': ' FORALL ',
    '\u2203': ' EXISTS ',
}


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Strip control characters (category "C*"), keep everything else
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')

    for a, b in _UNICODE_MAP.items():
        text = text.replace(a, b)
    for k, v in _LOGIC_MAP.items():
        text = text.replace(k, v)

    text = _URL_RE.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


def _is_noise(text: str) -> bool:
    t = text.strip()
    return (not t
            or bool(_SLIDE_NO.match(t))
            or bool(_URL_RE.fullmatch(t))
            or len(t) <= 2)


def _split_to_sentences(text: str) -> list:
    """Split on newlines first (pptx bullet separator), then on '. ' boundaries."""
    lines = [_clean(l) for l in text.split('\n')]
    lines = [l for l in lines if l and not _is_noise(l)]
    result = []
    for line in lines:
        sub = [s.strip() for s in line.split('. ') if s.strip()]
        result.extend(s for s in sub if not _is_noise(s))
    if result:
        return result
    c = _clean(text)
    return [c] if (c and not _is_noise(c)) else []


def _join_prose(parts: list) -> str:
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p[-1] not in '.!?':
            p += '.'
        out.append(p)
    return ' '.join(out)


def _deduplicate(items: list) -> list:
    seen, out = set(), []
    for item in items:
        key = item.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ── Picture alt-text ──────────────────────────────────────────────────────────

# Only alt-texts that contain real subject-matter keywords are kept.
# Decorative images (stock photos, logos, "thank you" slides) are discarded.
# NOTE: this keyword list was tuned for an IoT-security deck in the source
# notebook — you will likely want to broaden/replace it for your own content,
# or set use_ocr/alt-text filtering off entirely if it drops things you need.
_CONTENT_ALT = re.compile(
    r'(amazon|book|radar|chart|breakdown|threat|challenge|attack|'
    r'security|hack|password|protocol|botnet|malware|firmware|'
    r'statistics|infographic|case stud|nist|iot)',
    re.IGNORECASE
)


def _picture_alt(shape) -> Optional[str]:
    xml = etree.tostring(shape._element, pretty_print=True).decode()
    for raw in re.findall(r'descr="([^"]+)"', xml):
        cleaned = _clean(
            raw.replace('&amp;amp;', '&')
               .replace('&amp;', '&')
               .replace('&#8211;', '-')
               .replace('&#8220;', '"')
               .replace('&#8221;', '"')
        )
        if cleaned and not _is_noise(cleaned) and _CONTENT_ALT.search(cleaned):
            return cleaned
    return None


def split_into_passages(text: str, max_tokens: int = 300) -> list:
    """Greedy sentence-packing into ~max_tokens-word chunks (CoQA-sized passages)."""
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    passages = []
    current, current_len = [], 0

    for s in sentences:
        tok_len = len(s.split())
        if current and (current_len + tok_len > max_tokens):
            passages.append(" ".join(current))
            current, current_len = [], 0
        current.append(s)
        current_len += tok_len

    if current:
        passages.append(" ".join(current))

    return passages


# ── Core extractor ────────────────────────────────────────────────────────────

class SlideTextExtractor:
    def __init__(self, use_ocr: bool = False):
        self.use_ocr = use_ocr
        # Track alt-texts already used so repeated captions across slides are
        # only kept on the first slide they appear on.
        self._seen_alts: set = set()

    def extract_from_pptx(self, pptx_path: str) -> list:
        """
        Returns one dict per slide:
            slide_number, title, story (CoQA-ready prose),
            speaker_notes, raw_parts
        """
        self._seen_alts = set()
        prs = Presentation(pptx_path)
        results = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            title = ''
            body_parts = []
            img_parts = []

            for shape in slide.shapes:

                # Text frames (titles, content placeholders, text boxes)
                if shape.has_text_frame:
                    raw = shape.text_frame.text.strip()
                    if not raw or _is_noise(raw):
                        pass
                    elif self._is_title(shape):
                        title = _clean(raw)
                    else:
                        body_parts.extend(_split_to_sentences(raw))

                # Tables -> natural-language sentences
                elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                    body_parts.extend(self._table_sentences(shape))

                # Pictures -> alt-text (content-filtered, deck-deduplicated)
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    alt = _picture_alt(shape)
                    if alt:
                        key = alt.strip().lower()
                        if key not in self._seen_alts:
                            self._seen_alts.add(key)
                            img_parts.append(f"The image shows: {alt}.")

                    if self.use_ocr:
                        ocr_text = self._ocr(shape)
                        if ocr_text:
                            body_parts.extend(_split_to_sentences(ocr_text))

            # Speaker notes stored separately - NOT mixed into body_parts directly
            notes = ''
            if slide.has_notes_slide:
                raw_notes = slide.notes_slide.notes_text_frame.text.strip()
                c = _clean(raw_notes)
                if c and not _is_noise(c):
                    notes = c

            # Assemble story:
            # 1. Title  2. Body (skip if identical to title)  3. Image captions
            # 4. Speaker notes (split into sentences, folded in last)
            all_parts = []
            if title:
                all_parts.append(title)
            for p in body_parts:
                if p.lower().strip().rstrip('.') != title.lower().strip().rstrip('.'):
                    all_parts.append(p)
            all_parts.extend(img_parts)
            if notes:
                note_text = re.sub(r'([a-z])( +)([A-Z])', r'\1. \3', notes)
                all_parts.extend(_split_to_sentences(note_text))
            all_parts = _deduplicate(all_parts)

            results.append({
                'slide_number': slide_num,
                'title': title,
                'story': _join_prose(all_parts),
                'speaker_notes': notes,
                'raw_parts': all_parts,
            })

        return results

    def to_coqa_format(self, slides_data: list, filename: str) -> list:
        """Convert to CoQA JSON structure. Empty-story slides are excluded."""
        base = Path(filename).stem
        return [
            {
                'id': f"{base}_slide{s['slide_number']}",
                'filename': filename,
                'slide': s['slide_number'],
                'story': s['story'],
                'questions': [],
                'answers': [],
            }
            for s in slides_data if s['story'].strip()
        ]

    # ── private helpers ───────────────────────────────────────────────────────

    def _is_title(self, shape) -> bool:
        if not shape.is_placeholder:
            return False
        try:
            return shape.placeholder_format.type in (
                PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)
        except (AttributeError, ValueError):
            return False

    def _table_sentences(self, table_shape) -> list:
        rows = list(table_shape.table.rows)
        if not rows:
            return []
        headers = [_clean(c.text) for c in rows[0].cells if c.text.strip()]
        out = []
        for row in rows[1:]:
            cells = [_clean(c.text) for c in row.cells if c.text.strip()]
            if not cells:
                continue
            if headers and len(cells) == len(headers):
                out.append(', '.join(f"{h} is {v}" for h, v in zip(headers, cells)))
            else:
                out.append(' '.join(cells))
        return out

    def _ocr(self, image_shape) -> str:
        """
        OCR with preprocessing (only runs if use_ocr=True and pytesseract +
        the Tesseract binary are installed):
          - grayscale + contrast boost + 3x upscale for small text
          - fixes common 'loT' -> 'IoT' misread (lowercase L vs capital I)
          - PSM 11 (sparse text) works best for diagrams with scattered labels
          - returns only lines with 2+ real words to filter icon/symbol noise
        """
        try:
            import pytesseract
            from PIL import ImageEnhance
            img = Image.open(io.BytesIO(image_shape.image.blob))
            img = img.convert('L')
            img = ImageEnhance.Contrast(img).enhance(2.0)
            w, h = img.size
            img = img.resize((w * 3, h * 3), Image.LANCZOS)
            raw = pytesseract.image_to_string(img, config='--psm 11 --oem 3')
            raw = re.sub(r'\blo[Tt]\b', 'IoT', raw)
            raw = re.sub(r'\bIo[Tt]\b', 'IoT', raw)
            seen, clean_lines = set(), []
            for line in raw.split('\n'):
                words = re.findall(r'[A-Za-z&,]{2,}', line.strip())
                if len(words) >= 2:
                    joined = ' '.join(words)
                    key = joined.lower()
                    if key.startswith('of ') or key.startswith('and '):
                        continue
                    if key not in seen and len(joined) > 6:
                        seen.add(key)
                        clean_lines.append(joined)
            return '\n'.join(clean_lines)
        except Exception as e:
            logger.warning("OCR failed: %s", e)
            return ''


# ── Convenience runner ────────────────────────────────────────────────────────

def process_pptx(pptx_path: str,
                  use_ocr: bool = False,
                  save_json: bool = True,
                  output_dir: Optional[str] = None,
                  transcript_path: Optional[str] = None,
                  transcript_text: Optional[str] = None,
                  max_passage_tokens: int = 350,
                  verbose: bool = True) -> dict:
    """
    Runs the full preprocessing step on one .pptx file.

    Returns a dict with:
      - slides_data: per-slide dicts (slide_number, title, story, ...)
      - big_story: all slide stories (+ optional transcript) joined into one string
      - passages: big_story greedily chunked into ~max_passage_tokens-word pieces
      - coqa_slides: per-slide CoQA-format entries (empty stories excluded)
      - coqa_passages: passage-level CoQA-format entries

    If save_json=True, writes <stem>_coqa.json and <stem>_passages.json into
    output_dir (defaults to the current working directory).
    """
    extractor = SlideTextExtractor(use_ocr=use_ocr)
    slides_data = extractor.extract_from_pptx(pptx_path)

    # Optional transcript merge
    transcript = ""
    if transcript_path:
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        except Exception as e:
            print(f"Warning: Cannot read transcript file: {e}")
    if transcript_text:
        transcript = transcript_text.strip()
    if transcript:
        transcript = _clean(transcript)
        transcript = " ".join(_split_to_sentences(transcript))

    non_empty = [s for s in slides_data if s['story']]

    if verbose:
        print(f"Total slides : {len(slides_data)}")
        print(f"Non-empty    : {len(non_empty)}\n")
        for slide in non_empty:
            print(f"--- Slide {slide['slide_number']} ---")
            if slide['title']:
                print(f"Title : {slide['title']}")
            print(f"Story : {slide['story']}\n")

    big_story = " ".join(s["story"] for s in non_empty if s["story"])
    big_story = re.sub(r'\s+', ' ', big_story).strip()
    if transcript:
        big_story = re.sub(r'\s+', ' ', (big_story + " " + transcript)).strip()

    passages = split_into_passages(big_story, max_tokens=max_passage_tokens)

    if verbose:
        print("\n=========== AUTO-GENERATED COQA PASSAGES ===========\n")
        for i, p in enumerate(passages, 1):
            print(f"[PASSAGE {i}] ({len(p.split())} words)")
            print(p)
            print("\n-----------------------------------------------------\n")

    stem = Path(pptx_path).stem
    coqa_passages = [
        {
            "id": f"{stem}_passage{i}",
            "filename": pptx_path,
            "slide": None,
            "story": p,
            "questions": [],
            "answers": [],
        }
        for i, p in enumerate(passages, 1)
    ]
    coqa_slides = extractor.to_coqa_format(slides_data, pptx_path)

    if save_json:
        out_dir = Path(output_dir) if output_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        passages_out = out_dir / f"{stem}_passages.json"
        with open(passages_out, "w", encoding="utf-8") as f:
            json.dump({"data": coqa_passages}, f, indent=2, ensure_ascii=False)

        slides_out = out_dir / f"{stem}_coqa.json"
        with open(slides_out, "w", encoding="utf-8") as f:
            json.dump({"data": coqa_slides}, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"Saved {len(passages)} passages -> {passages_out}")
            print(f"Saved {len(coqa_slides)} slide entries -> {slides_out}")

    return {
        "slides_data": slides_data,
        "big_story": big_story,
        "passages": passages,
        "coqa_slides": coqa_slides,
        "coqa_passages": coqa_passages,
    }
