"""
Auto-converted from main.ipynb with minimal changes to make functions importable and runnable.

Notes:
- Notebook 'execution' cells were wrapped into callable functions (run_stage1/2/3, run_full_pipeline).
- Dependency installation via pip was removed; install requirements in your environment instead.
- Gemini API key is read from env var GEMINI_API_KEY (or passed explicitly) instead of being hard-coded.
"""

from __future__ import annotations


import json
import numpy as np
import os
import re

# Third-party libraries
try:
    import tf_keras  # type: ignore
except Exception:
    tf_keras = None  # type: ignore

# UPDATED: New import syntax for the new google-genai package
try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore


# Optional: Suppress the TensorFlow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
try:
    import tensorflow as tf  # type: ignore
    tf.get_logger().setLevel('ERROR')
except Exception:
    tf = None  # type: ignore


GEMINI_CLIENT = None


def configure_gemini(api_key: str | None = None):
    """Initialize the global Gemini client.

    Args:
        api_key: If None, reads from environment variable GEMINI_API_KEY.

    Returns:
        The initialized client, or None if api_key is missing/invalid.
    """
    global GEMINI_CLIENT
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        GEMINI_CLIENT = None
        return None
    if genai is None:
        GEMINI_CLIENT = None
        return None
    try:
        GEMINI_CLIENT = genai.Client(api_key=key)
        return GEMINI_CLIENT
    except Exception:
        GEMINI_CLIENT = None
        return None


# ---- Notebook cell 2 ----
def transcribe_audio(audio_path, model_size="base", device="cpu", compute_type="int8"):
    if WhisperModel is None:
        raise ImportError("faster-whisper is required for transcribe_audio. Install it with: pip install faster-whisper")
    """
    Transcribe audio file using faster-whisper (CTranslate2) with word-level timestamps.
    
    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v2", "large-v3")
        device: Device to use ("cpu", "cuda")
        compute_type: Computation type ("int8", "int8_float16", "float16", "float32")
    
    Returns:
        dict with keys: 'text', 'segments', 'language', 'language_probability', 'duration'
    """
    print(f"Loading faster-whisper model: {model_size}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print(f"Transcribing: {audio_path}...")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    # Convert faster-whisper output to dict format compatible with extract functions
    segments_list = []
    full_text_parts = []
    
    for segment in segments:
        segment_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": []
        }
        
        # Extract words with timestamps
        if hasattr(segment, 'words') and segment.words:
            for word_obj in segment.words:
                segment_dict["words"].append({
                    "word": word_obj.word,
                    "start": word_obj.start,
                    "end": word_obj.end,
                })
        
        segments_list.append(segment_dict)
        full_text_parts.append(segment.text.strip())
    
    result = {
        "text": " ".join(full_text_parts),
        "segments": segments_list,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }
    
    return result


# ---- Notebook cell 3 ----
def detect_filler_words_per_sentence(sentences):
    """
    Detect and count filler words in each sentence.
    
    Args:
        sentences: List of sentence strings
    
    Returns:
        dict with 'filler_words_per_sentence', 'total_filler_words', 'summary'
    """
    # Common filler words and phrases (case-insensitive matching)
    # Single-word fillers
    single_word_fillers = [
        "um", "uh", "er", "ah", "hmm",
        "like", "so", "well", "actually", "basically",
        "right", "okay", "ok",
        "literally", "honestly", "obviously", "probably"
    ]
    
    # Multi-word filler phrases
    multi_word_fillers = [
        "you know", "I mean", "sort of", "kind of",
        "you see", "I guess", "I think", "you know what"
    ]
    
    filler_counts = []
    total_filler_words = 0
    
    for i, sentence in enumerate(sentences):
        count = 0
        sentence_lower = sentence.lower()
        
        # Count single-word fillers (with word boundaries)
        for filler in single_word_fillers:
            pattern = rf'\b{re.escape(filler)}\b'
            count += len(re.findall(pattern, sentence_lower))
        
        # Count multi-word fillers (as phrases)
        for filler in multi_word_fillers:
            pattern = rf'\b{re.escape(filler)}\b'
            count += len(re.findall(pattern, sentence_lower))
        filler_counts.append({
            "sentence_index": i,
            "sentence": sentence,
            "filler_word_count": count,
        })
        total_filler_words += count
    
    # Summary statistics
    counts_only = [item["filler_word_count"] for item in filler_counts]
    summary = {
        "total_filler_words": total_filler_words,
        "total_sentences": len(sentences),
        "mean_filler_words_per_sentence": round(sum(counts_only) / len(counts_only), 2) if counts_only else 0.0,
        "max_filler_words_in_sentence": max(counts_only) if counts_only else 0,
        "sentences_with_filler_words": sum(1 for c in counts_only if c > 0),
    }
    
    return {
        "feature": "filler_word_analysis",
        #"filler_words_per_sentence": filler_counts,
        "summary": summary,
    }


# ---- Notebook cell 4 ----
def calculate_wpm(words):
    """
    Calculate Words Per Minute (WPM) from word-level timestamps.
    
    Args:
        words: List of dicts with 'word', 'start', 'end' keys
    
    Returns:
        dict with 'wpm', 'total_words', 'duration_seconds', 'duration_minutes'
    """
    if not words:
        return {
            "wpm": 0.0,
            "total_words": 0,
            "duration_seconds": 0.0,
            "duration_minutes": 0.0,
        }
    
    total_words = len(words)
    first_word_start = words[0]["start"]
    last_word_end = words[-1]["end"]
    duration_seconds = last_word_end - first_word_start
    duration_minutes = duration_seconds / 60.0
    
    # Calculate WPM: (total_words / duration_in_minutes)
    wpm = (total_words * 60.0) / duration_seconds if duration_seconds > 0 else 0.0
    
    return {
        "wpm": round(wpm, 2),
        "total_words": total_words,
        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_minutes, 2),
    }


# ---- Notebook cell 5 ----
def extract_words_from_whisper(whisper_result):
    """
    Extract word-level timestamps from Whisper output.
    Returns list of {word, start, end} matching DUMMY_WORDS format.
    """
    words = []
    
    # Whisper segments contain words with timestamps
    for segment in whisper_result.get("segments", []):
        # Check if segment has words attribute
        if "words" in segment:
            for word_info in segment["words"]:
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0.0),
                    "end": word_info.get("end", 0.0),
                })
        # Fallback: if no word-level timestamps, approximate from segment
        elif "start" in segment and "end" in segment:
            # Split segment text into words (rough approximation)
            segment_text = segment.get("text", "").strip()
            segment_words = segment_text.split()
            if segment_words:
                duration = segment["end"] - segment["start"]
                word_duration = duration / len(segment_words)
                for i, word in enumerate(segment_words):
                    words.append({
                        "word": word,
                        "start": segment["start"] + i * word_duration,
                        "end": segment["start"] + (i + 1) * word_duration,
                    })
    
    return words


# ---- Notebook cell 6 ----
def extract_sentences_from_whisper(whisper_result):
    """
    Extract sentences from Whisper transcription.
    Splits full text by sentence-ending punctuation (. ! ?).
    """
    full_text = whisper_result.get("text", "").strip()
    
    if not full_text:
        return []
    
    # Split by sentence-ending punctuation, keeping the punctuation
    # Pattern: split on . ! ? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    # Clean up: remove empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


# ---- Notebook cell 8 ----
def find_pause_gaps(words, min_gap_sec=0.75):
    """Find silences ≥ min_gap_sec between consecutive words."""
    gaps = []
    for i in range(len(words) - 1):
        gap_start = words[i]["end"]
        gap_end = words[i + 1]["start"]
        duration = gap_end - gap_start
        if duration >= min_gap_sec:
            gaps.append({
                "gap_start": gap_start,
                "gap_end": gap_end,
                "duration": duration,
                "word_before": words[i]["word"].strip(),
                "word_after": words[i + 1]["word"].strip(),
            })
    return gaps


def is_functional_pause(word_before: str) -> bool:
    """Punctuation before silence → functional pause."""
    return word_before.endswith(".") or word_before.endswith(",") or word_before.endswith("!") or word_before.endswith("?") or word_before.endswith(":") or word_before.endswith(";")


def _norm(w):
    return w.lower().strip().rstrip(".,;:!?")


def is_stutter_recovery(word_before: str, word_after: str) -> bool:
    """True if word_after repeats word_before (stuttering)."""
    return _norm(word_before) == _norm(word_after)


# ---- Notebook cell 9 ----
def run_advanced_pause_analysis(words, min_gap_sec=0.75):
    """Full Feature A pipeline → JSON for Stage 2."""
    gaps = find_pause_gaps(words, min_gap_sec)
    functional = []
    dysfunctional = []

    for g in gaps:
        w_before, w_after = g["word_before"], g["word_after"]
        stutter = is_stutter_recovery(w_before, w_after)
        base = {
            "gap_start": g["gap_start"],
            "gap_end": g["gap_end"],
            "duration": g["duration"],
            "word_before": w_before,
            "word_after": w_after,
            "stutter_recovery": stutter,
        }

        if is_functional_pause(w_before):
            functional.append(base)
        else:
            dysfunctional.append(base)

    out = {
        "feature": "advanced_pause_analysis",
        "pause_gaps_total": len(gaps),
        "pause_gaps_over_0_75s": len(gaps),
        "functional_pauses": functional,
        "dysfunctional_pauses": dysfunctional,
        "summary": {
            "count_functional": len(functional),
            "count_dysfunctional": len(dysfunctional),
            "count_stutter_recovery": sum(1 for x in functional + dysfunctional if x["stutter_recovery"]),
        },
    }
    return out


# ---- Notebook cell 11 ----
model = None  # Lazy init; created inside run_stage1_feature_extraction
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_intro_embedding(embeddings):
    n = min(3, len(embeddings))
    return np.mean(embeddings[:n], axis=0)


def run_structure_evaluation(sentences):
    """Full Feature B pipeline → JSON for Stage 2."""
    global model
    if model is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for structure evaluation. Install it with: pip install sentence-transformers")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if len(sentences) < 2:
        return {
            "feature": "structure_evaluation",
            "num_sentences": len(sentences),
            "flow_scores": [],
            "flow_breaks": [],
            "drift_scores": [],
            "summary": {},
        }

    emb = model.encode(sentences)
    flow_scores = []
    flow_breaks = []

    for i in range(len(emb) - 1):
        sim = cosine_similarity(emb[i], emb[i + 1])
        flow_scores.append({"sentence_i": i, "sentence_j": i + 1, "cosine_similarity": sim})
        if sim < 0.35:
            flow_breaks.append({"between": [i, i + 1], "cosine_similarity": sim})

    intro_emb = get_intro_embedding(emb)
    drift_scores = []
    for j in range(3, len(emb)):
        sim = cosine_similarity(intro_emb, emb[j])
        drift_scores.append({"sentence_index": j, "cosine_similarity_to_intro": sim})

    flows = [x["cosine_similarity"] for x in flow_scores]
    drifts = [x["cosine_similarity_to_intro"] for x in drift_scores]
    summary = {
        "mean_flow_similarity": round(float(np.mean(flows)), 4) if flows else None,
        "flow_break_count": len(flow_breaks),
        "mean_drift_similarity": round(float(np.mean(drifts)), 4) if drifts else None,
        "min_drift_similarity": round(float(np.min(drifts)), 4) if drifts else None,
    }

    return {
        "feature": "structure_evaluation",
        "num_sentences": len(sentences),
        "flow_scores": flow_scores,
        "flow_breaks": flow_breaks,
        "drift_scores": drift_scores,
        "summary": summary,
    }


# ---- Notebook cell 13 ----
def filter_contentful_sentences(sentences, min_tokens=5, filler_patterns=None):
    """
    Filter out very short or filler-only sentences before embedding.
    filtering them reduces noise and focuses analysis on contentful sentences.
    
    Args:
        sentences: List of sentence strings
        min_tokens: Minimum number of tokens (words) to keep
        filler_patterns: List of filler words/phrases to exclude (optional)
    
    Returns:
        filtered_sentences: List of sentences that pass the filter
        original_indices: List mapping filtered index -> original index
    """
    if filler_patterns is None:
        filler_patterns = ["ok", "okay", "right", "um", "uh", "er", "ah", "hmm"]
    
    filtered = []
    original_indices = []
    
    for i, sent in enumerate(sentences):
        # Tokenize roughly (split on whitespace)
        tokens = sent.strip().split()
        
        # Skip if too short
        if len(tokens) < min_tokens:
            continue
        
        # Skip if it's just filler words (optional check)
        sent_lower = sent.lower().strip()
        if sent_lower in filler_patterns or len(tokens) <= 2:
            # Check if it's mostly filler
            is_filler = all(token.lower().rstrip(".,!?") in filler_patterns for token in tokens)
            if is_filler:
                continue
        
        filtered.append(sent)
        original_indices.append(i)
    
    return filtered, original_indices


def normalize_embeddings(embeddings, eps=1e-8):
    """
    L2-normalize embeddings for numeric stability.
    Normalizing makes Dot Product == Cosine Similarity.
    Args:
        embeddings: numpy array of shape (n_sentences, embedding_dim)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return embeddings / norms  


def smooth_similarities(similarities, window_size=3):
    """
    Apply moving average smoothing to similarity scores.
    producing a cleaner similarity curve where local minima/peaks are more meaningful.
    Args:
        similarities: List or array of similarity scores
        window_size: Size of moving average window (should be odd)
    
    Returns:
        smoothed similarities
    """
    if len(similarities) == 0:
        return similarities
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(similarities)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(similarities), i + half_window + 1)
        window_vals = similarities[start_idx:end_idx]
        smoothed.append(np.mean(window_vals))
    
    return np.array(smoothed)


def compute_depth_scores(similarities, window_size=3):
    """
    Compute depth scores for each position (how deep is local minimum relative to peaks).
    
    Args:
        similarities: Smoothed similarity scores
        window_size: Size of window to look for peaks on left/right
    
    Returns:
        depth_scores: Array of depth values (higher = deeper local minimum)
    """
    n = len(similarities)
    depth_scores = np.zeros(n)
    half_window = window_size // 2
    
    for i in range(n):
        # Find peak on the left
        left_start = max(0, i - half_window)
        left_peak = np.max(similarities[left_start:i]) if i > left_start else similarities[i]
        
        # Find peak on the right
        right_end = min(n, i + half_window + 1)
        right_peak = np.max(similarities[i+1:right_end]) if i+1 < right_end else similarities[i]
        
        # Depth = average of peaks - current value
        avg_peak = (left_peak + right_peak) / 2.0
        depth_scores[i] = max(0, avg_peak - similarities[i])
    
    return depth_scores


def find_local_minima(similarities, depth_scores, min_depth=None):
    """
    Find positions that are local minima in smoothed similarities.
    
    Args:
        similarities: Smoothed similarity scores
        depth_scores: Depth scores for each position
        min_depth: Optional minimum depth threshold
    
    Returns:
        List of indices that are local minima
    """
    minima = []
    
    for i in range(1, len(similarities) - 1):
        # Check if it's a local minimum
        if similarities[i] < similarities[i-1] and similarities[i] < similarities[i+1]:
            if min_depth is None or depth_scores[i] >= min_depth:
                minima.append(i)
    
    return minima


def get_intro_centroid_filtered(embeddings, sentences, min_intro_tokens=8, n_intro_sentences=3):
    """
    Build intro centroid from first few contentful sentences (filtered).
    
    Args:
        embeddings: Normalized embeddings array
        sentences: Original sentence list
        min_intro_tokens: Minimum tokens for a sentence to be considered contentful
        n_intro_sentences: Target number of intro sentences
    
    Returns:
        intro_centroid: Normalized embedding vector
    """
    intro_indices = []
    
    for i, sent in enumerate(sentences[:min(len(sentences), n_intro_sentences * 2)]):
        tokens = sent.strip().split()
        if len(tokens) >= min_intro_tokens:
            intro_indices.append(i)
            if len(intro_indices) >= n_intro_sentences:
                break
    
    if not intro_indices:
        # Fallback: use first few sentences regardless
        intro_indices = list(range(min(n_intro_sentences, len(embeddings))))
    
    # Average the intro embeddings
    intro_emb = np.mean(embeddings[intro_indices], axis=0)
    
    # Re-normalize
    norm = np.linalg.norm(intro_emb)
    if norm > 1e-8:
        intro_emb = intro_emb / norm
    
    return intro_emb


# ---- Notebook cell 14 ----
def run_structure_evaluation_improved(sentences, 
                                       model=None,
                                       min_tokens=5,
                                       smoothing_window=3,
                                       depth_window=3,
                                       depth_alpha=0.8,
                                       hard_threshold=0.25,
                                       drift_threshold=0.4,
                                       min_intro_tokens=8,
                                       n_intro_sentences=3):
    """
    Improved Feature B pipeline with depth-based segmentation and robust filtering.
    
    Args:
        sentences: List of sentence strings
        model: SentenceTransformer model (if None, creates new one)
        min_tokens: Minimum tokens to keep a sentence
        smoothing_window: Window size for moving average smoothing
        depth_window: Window size for depth computation
        depth_alpha: Multiplier for std deviation in dynamic threshold (mean + alpha*std)
        hard_threshold: Hard fallback threshold for flow breaks (< this = definitely a break)
        drift_threshold: Threshold for drift detection (sim < this = off-topic)
        min_intro_tokens: Minimum tokens for intro sentence
        n_intro_sentences: Target number of intro sentences
    
    Returns:
        dict with improved structure evaluation metrics
    """
    if model is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for structure evaluation. Install it with: pip install sentence-transformers")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    
    if len(sentences) < 2:
        return {
            "feature": "structure_evaluation_improved",
            "num_sentences": len(sentences),
            "num_filtered_sentences": 0,
            "flow": {},
            "drift": {},
            "summary": {},
        }
    
    # Step 1: Filter out very short/filler sentences
    filtered_sentences, original_indices = filter_contentful_sentences(sentences, min_tokens=min_tokens)
    
    if len(filtered_sentences) < 2:
        # Fallback: use all sentences if filtering removes too many
        filtered_sentences = sentences
        original_indices = list(range(len(sentences)))
    
    # Step 2: Encode and normalize embeddings
    embeddings_raw = model.encode(filtered_sentences)
    embeddings = normalize_embeddings(embeddings_raw, eps=1e-8)
    
    # Step 3: Compute adjacent similarities
    n = len(embeddings)
    sim_adj_raw = []
    for i in range(n - 1):
        # Since embeddings are normalized, cosine = dot product
        sim = float(np.dot(embeddings[i], embeddings[i+1]))
        sim_adj_raw.append(sim)
    
    sim_adj_raw = np.array(sim_adj_raw)
    
    # Step 4: Smooth similarities
    sim_adj_smooth = smooth_similarities(sim_adj_raw, window_size=smoothing_window)
    
    # Step 5: Compute depth scores
    depth_scores = compute_depth_scores(sim_adj_smooth, window_size=depth_window)
    
    # Step 6: Find segment boundaries using depth-based detection
    # Dynamic threshold: mean + alpha * std
    mean_depth = np.mean(depth_scores)
    std_depth = np.std(depth_scores)
    dynamic_threshold = mean_depth + depth_alpha * std_depth
    
    # Find local minima
    local_minima = find_local_minima(sim_adj_smooth, depth_scores, min_depth=None)
    
    # Filter minima by depth threshold OR hard similarity threshold
    segment_boundaries = []
    flow_breaks = []
    
    for i in local_minima:
        # Check if it passes depth threshold OR hard similarity threshold
        if depth_scores[i] >= dynamic_threshold or sim_adj_smooth[i] < hard_threshold:
            # Map back to original sentence indices
            orig_i = original_indices[i] if i < len(original_indices) else i
            orig_j = original_indices[i+1] if i+1 < len(original_indices) else i+1
            
            segment_boundaries.append({
                "boundary_index": i,
                "original_sentence_indices": [orig_i, orig_j],
                "cosine_similarity": float(sim_adj_smooth[i]),
                "depth_score": float(depth_scores[i]),
                "detection_method": "depth" if depth_scores[i] >= dynamic_threshold else "hard_threshold"
            })
            
            flow_breaks.append({
                "between": [orig_i, orig_j],
                "cosine_similarity": float(sim_adj_smooth[i]),
                "depth_score": float(depth_scores[i])
            })
    
    # Step 7: Compute drift scores with filtered intro
    intro_centroid = get_intro_centroid_filtered(
        embeddings, filtered_sentences, 
        min_intro_tokens=min_intro_tokens,
        n_intro_sentences=n_intro_sentences
    )
    
    drift_scores = []
    drift_sentences = []
    
    for j in range(len(embeddings)):
        sim_to_intro = float(np.dot(intro_centroid, embeddings[j]))
        orig_j = original_indices[j] if j < len(original_indices) else j
        
        drift_scores.append({
            "sentence_index": orig_j,
            "filtered_index": j,
            "cosine_similarity_to_intro": sim_to_intro
        })
        
        if sim_to_intro < drift_threshold:
            drift_sentences.append({
                "sentence_index": orig_j,
                "filtered_index": j,
                "cosine_similarity_to_intro": sim_to_intro
            })
    
    # Step 8: Summary statistics
    summary = {
        "num_original_sentences": len(sentences),
        "num_filtered_sentences": len(filtered_sentences),
        "filtering_ratio": round(len(filtered_sentences) / len(sentences), 3) if sentences else 0.0,
        "mean_flow_similarity_raw": round(float(np.mean(sim_adj_raw)), 4) if len(sim_adj_raw) > 0 else None,
        "mean_flow_similarity_smooth": round(float(np.mean(sim_adj_smooth)), 4) if len(sim_adj_smooth) > 0 else None,
        "std_flow_similarity": round(float(np.std(sim_adj_smooth)), 4) if len(sim_adj_smooth) > 0 else None,
        "segment_boundary_count": len(segment_boundaries),
        "mean_depth_score": round(float(np.mean(depth_scores)), 4) if len(depth_scores) > 0 else None,
        "dynamic_threshold_depth": round(float(dynamic_threshold), 4),
        "mean_drift_similarity": round(float(np.mean([d["cosine_similarity_to_intro"] for d in drift_scores])), 4) if drift_scores else None,
        "min_drift_similarity": round(float(np.min([d["cosine_similarity_to_intro"] for d in drift_scores])), 4) if drift_scores else None,
        "fraction_sentences_off_topic": round(len(drift_sentences) / len(drift_scores), 3) if drift_scores else 0.0,
        "drift_sentence_count": len(drift_sentences),
    }
    
    return {
        "feature": "structure_evaluation_improved",
        "num_sentences": len(sentences),
        "num_filtered_sentences": len(filtered_sentences),
        "flow": {
            "adjacent_similarity_raw": [float(x) for x in sim_adj_raw],
            "adjacent_similarity_smooth": [float(x) for x in sim_adj_smooth],
            "depth_scores": [float(x) for x in depth_scores],
            "segment_boundaries": segment_boundaries,
            "flow_breaks": flow_breaks,
            "dynamic_threshold_depth": float(dynamic_threshold),
            "hard_threshold": hard_threshold,
        },
        "drift": {
            "similarity_to_intro": [d["cosine_similarity_to_intro"] for d in drift_scores],
            "drift_sentences": drift_sentences,
            "drift_threshold": drift_threshold,
        },
        "summary": summary,
    }


# ---- Notebook cell 17 ----
try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore

import numpy as np
import json

def load_audio(file_path):
    """Loads audio and returns signal, sample rate."""
    if librosa is None:
        raise ImportError("librosa is required for audio quality analysis. Install it with: pip install librosa")
    try:
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def calculate_frame_snr(current_rms, global_rms, percentile=10):
    """Calculates SNR for a single frame relative to the global noise floor."""
    noise_floor = np.percentile(global_rms, percentile)
    if current_rms <= 0 or noise_floor <= 0:
        return 0.0
    return float(20 * np.log10(current_rms / noise_floor))

def process_audio_per_frame(file_path, output_json="frame_analysis.json"):
    y, sr = load_audio(file_path)
    if y is None: return

    hop_length = 512
    
    # 1. Extract Pitch for variance logic (internal use only)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=hop_length)
    
    # 2. Extract Loudness/RMS
    rms_array = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    loudness_spl = librosa.amplitude_to_db(rms_array, ref=np.max) + 100

    results = []
    window_size = 5 

    for i in range(len(f0)):
        # Calculate Local Pitch SD (Internal logic for the status)
        start_win = max(0, i - window_size)
        end_win = min(len(f0), i + window_size)
        local_window = f0[start_win:end_win]
        local_window = local_window[local_window > 0] 
        
        pitch_sd = np.std(local_window) if len(local_window) > 0 else 0.0
        
        # Determine "Pitch Variance" status
        if f0[i] <= 0 or np.isnan(f0[i]):
            p_variance = "SILENT"
        else:
            p_variance = "LOW" if pitch_sd < 15 else "HIGH"

        # Calculate SNR for this frame
        snr_val = calculate_frame_snr(rms_array[i], rms_array)

        # Build clean dictionary
        frame_data = {
            "frame_index": i,
            "loudness_db": round(float(max(0, loudness_spl[i])), 2),
            "snr_db": round(float(max(0, snr_val)), 2),
            "pitch_variance": p_variance
        }
        results.append(frame_data)

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Data saved to {output_json}")




# ---- Notebook cell 19 ----
def run_analysis_agent(feature_extraction_json, metrics_json_path, model_name="gemini-2.5-flash"):
    """Run the Gemini-based Analysis Agent for Stage 2 using google-genai.

    Args:
        feature_extraction_json: dict produced by Stage 1 (`raw_data_json`).
        metrics_json_path: path to `metrics.json` with thresholds.
        model_name: Gemini model name (e.g., "gemini-2.5-flash").

    Returns:
        dict: Parsed JSON analysis from Gemini, or an error object.
    """
    if GEMINI_CLIENT is None:
        return {
            "error": "Gemini client not initialized",
            "detail": "Ensure GEMINI_API_KEY is set and the updated Gemini config cell has been run."
        }

    # Load metrics thresholds
    try:
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            metrics_thresholds = json.load(f)
    except FileNotFoundError:
        return {
            "error": "metrics.json not found",
            "detail": f"Could not find metrics.json at {metrics_json_path}"
        }

    # Persona + task instructions
    persona_instructions = (
        "You are the Analysis Agent. Your role is to read the JSON metrics and interpret them objectively.\n\n"
        "Tasks:\n"
        "1. Normalize each metric using standard communication science thresholds.\n"
        "2. Identify the top 3 presentation issues.\n"
        "3. Write a structured summary using Observation-Impact-Suggestion (OIS) labels BUT do not write full feedback.\n"
        "4. Output a CLEAN JSON object with:\n"
        "   - observations\n"
        "   - impacts\n"
        "   - suggestions (short phrases, not full sentences)\n"
        "   - reasoning\n"
        "   - confidence scores\n"
        "5. Do NOT include emotional tone or praise. You are purely analytical.\n\n"
    )

    # Expected output schema (for the model to follow)
    schema_instructions = (
        "Return ONLY valid JSON with this shape (no markdown, no extra text):\n"
        "{\n"
        "  \"top_3_issues\": [\n"
        "    {\"issue\": \"...\", \"severity\": \"...\", \"category\": \"...\"}\n"
        "  ],\n"
        "  \"observations\": [\n"
        "    {\"metric\": \"...\", \"value\": \"...\", \"threshold\": \"...\", \"status\": \"...\", \"label\": \"Observation\"}\n"
        "  ],\n"
        "  \"impacts\": [\n"
        "    {\"issue\": \"...\", \"impact_description\": \"...\", \"label\": \"Impact\"}\n"
        "  ],\n"
        "  \"suggestions\": [\n"
        "    \"short phrase 1\",\n"
        "    \"short phrase 2\"\n"
        "  ],\n"
        "  \"reasoning\": \"short explanation of how you ranked issues and mapped metrics\",\n"
        "  \"confidence_scores\": {\n"
        "    \"overall_confidence\": 0.0,\n"
        "    \"metric_confidence\": { \"metric_name\": 0.0 }\n"
        "  }\n"
        "}\n\n"
        "Values should be concise and machine-readable. Do not include any text outside the JSON object."
    )

    # Build the full prompt
    prompt = (
        persona_instructions
        + schema_instructions
        + "\nFeature Extraction Data (from Stage 1, raw_data_json):\n"
        + json.dumps(feature_extraction_json, indent=2)
        + "\n\nMetrics Thresholds (metrics.json):\n"
        + json.dumps(metrics_thresholds, indent=2)
    )

    # Call Gemini via the google-genai client
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=model_name,
            contents=prompt,
        )
    except Exception as e:
        return {
            "error": "Gemini API call failed",
            "detail": str(e),
        }

    # Extract text
    response_text = getattr(response, "text", None)
    if not response_text:
        response_text = str(response)
    response_text = response_text.strip()

    # Strip optional markdown fences
    if "```json" in response_text:
        response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()

    # Parse JSON
    try:
        analysis_result = json.loads(response_text)
        print("Analysis completed successfully.")
        return analysis_result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response from Gemini: {e}")
        return {
            "error": "Failed to parse JSON response from Gemini",
            "detail": str(e),
            "raw_response": response_text[:2000],
        }


# ---- Notebook cell 21 ----
def run_feedback_agent(analysis_json, model_name="gemini-2.5-flash"):
    """Run the Gemini-based Feedback Agent for Stage 3.

    Args:
        analysis_json: dict produced by Stage 2 (`analysis_result_v2`).
        model_name: Gemini model name.

    Returns:
        dict: Parsed JSON or text output from the Feedback Agent.
    """
    if GEMINI_CLIENT is None:
        return {
            "error": "Gemini client not initialized",
            "detail": "Ensure GEMINI_API_KEY is set."
        }

    persona_instructions = (
        "You are a world-class Public Speaking Coach. Your style is warm, supportive, and specific. "
        "Using the analysis provided, write high-quality feedback using:\n\n"
        "1. The Sandwich Method: Genuine praise, a constructive OIS section, and motivational encouragement.\n"
        "2. Observation-Impact-Suggestion (OIS) for each issue: Clear observation from the data, "
        "psychological impact on the audience, and one actionable, concrete improvement.\n"
        "3. Limit suggestions to the TOP THREE issues only.\n"
        "4. NEVER contradict the analysis JSON.\n"
        "5. Write as if speaking to a student who is anxious but eager to improve.\n"
        "6. Don't use emojis.\n\n"
        "Return the output as a CLEAN JSON object with the following keys:\n"
        "- 'praise': string\n"
        "- 'constructive_feedback': list of { 'issue': string, 'ois_delivery': string }\n"
        "- 'encouragement': string\n"
        "- 'full_formatted_text': string (the complete feedback combined)\n"
    )

    # Build the prompt using Stage 2 output
    prompt = (
        persona_instructions
        + "\nAnalysis Data (from Stage 2):\n"
        + json.dumps(analysis_json, indent=2)
    )

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=model_name,
            contents=prompt,
        )
    except Exception as e:
        return {
            "error": "Gemini API call failed",
            "detail": str(e),
        }

    response_text = getattr(response, "text", None) or str(response)
    response_text = response_text.strip()

    # Clean markdown formatting if present
    if "```json" in response_text:
        response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        feedback_result = json.loads(response_text)
        print("Feedback generation completed successfully.")
        return feedback_result
    except json.JSONDecodeError:
        # If the prompt expects a non-JSON string, return as a dict wrapper
        return {"content": response_text}


# =====================
# Wrapped notebook execution
# =====================

def run_stage1_feature_extraction(
    audio_path: str,
    whisper_model_size: str = "base",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    min_pause_gap_sec: float = 0.75,
    *,
    structure_model_name: str = "all-MiniLM-L6-v2",
    structure_params: dict | None = None,
) -> dict:
    """Stage 1: Transcribe audio, compute features, and return raw_data_json."""
    # Transcribe + parse
    whisper_result = transcribe_audio(
        audio_path,
        model_size=whisper_model_size,
        device=whisper_device,
        compute_type=whisper_compute_type,
    )
    words = extract_words_from_whisper(whisper_result)
    sentences = extract_sentences_from_whisper(whisper_result)

    # Feature: WPM
    wpm_result = calculate_wpm(words)

    # Feature: filler words
    filler_words_output = detect_filler_words_per_sentence(sentences)

    # Feature A: pause analysis
    feature_a_output = run_advanced_pause_analysis(words, min_gap_sec=min_pause_gap_sec)

    # Feature B: structure evaluation (improved)
    # Ensure the embedding model exists.
    global model
    try:
        model  # type: ignore[name-defined]
    except Exception:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for structure evaluation. Install it with: pip install sentence-transformers")
        model = SentenceTransformer(structure_model_name)

    params = structure_params or {}
    feature_b_improved_output = run_structure_evaluation_improved(
        sentences,
        model=model,
        **params,
    )

    raw_data_json = {
        "advanced_pause_analysis": feature_a_output,
        "structure_evaluation": feature_b_improved_output,
        "words_per_minute": wpm_result,
        "filler_word_analysis": filler_words_output,
    }
    return raw_data_json


def run_stage2_analysis(
    raw_data_json: dict,
    metrics_json_path: str,
    *,
    model_name: str = "gemini-2.5-flash",
    gemini_api_key: str | None = None,
) -> dict:
    """Stage 2: Run Gemini analysis agent."""
    if GEMINI_CLIENT is None:
        configure_gemini(gemini_api_key)
    return run_analysis_agent(raw_data_json, metrics_json_path, model_name=model_name)


def run_stage3_feedback(
    analysis_json: dict,
    *,
    model_name: str = "gemini-2.5-flash",
    gemini_api_key: str | None = None,
) -> dict:
    """Stage 3: Run Gemini feedback agent."""
    if GEMINI_CLIENT is None:
        configure_gemini(gemini_api_key)
    return run_feedback_agent(analysis_json, model_name=model_name)


def run_full_pipeline(
    audio_path: str,
    metrics_json_path: str,
    *,
    whisper_model_size: str = "base",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    gemini_model_name: str = "gemini-2.5-flash",
    gemini_api_key: str = "AIzaSyD08Ido5tImuVlBldgdg6bybr2BVC1Sxhk",
) -> dict:
    """Run Stage 1 -> Stage 2 -> Stage 3 in order and return outputs."""
    raw_data = run_stage1_feature_extraction(
        audio_path,
        whisper_model_size=whisper_model_size,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
    )

    analysis = run_stage2_analysis(
        raw_data,
        metrics_json_path,
        model_name=gemini_model_name,
        gemini_api_key=gemini_api_key,
    )

    feedback = None
    if analysis is not None and isinstance(analysis, dict) and "error" not in analysis:
        feedback = run_stage3_feedback(
            analysis,
            model_name=gemini_model_name,
            gemini_api_key=gemini_api_key,
        )

    return {"stage1_raw_data": raw_data, "stage2_analysis": analysis, "stage3_feedback": feedback}