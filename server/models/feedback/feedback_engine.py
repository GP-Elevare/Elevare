import os
import re
import json
import numpy as np
import librosa
from google import genai
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
import tensorflow as tf
from pydub import AudioSegment
import pyloudnorm as pyln

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Global Client Placeholder
GEMINI_CLIENT = None
SENTENCE_MODEL = None


def initialize_models(api_key=None):
    """Initializes the Sentence Transformer and Gemini Client."""
    global GEMINI_CLIENT, SENTENCE_MODEL
    
    # Load Sentence Transformer (loads once)
    if SENTENCE_MODEL is None:
        print("Loading SentenceTransformer model...")
        SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    # Configure Gemini
    if api_key:
        try:
            GEMINI_CLIENT = genai.Client(api_key=api_key)
            print("Gemini Client configured successfully.")
        except Exception as e:
            print(f"Error initializing Gemini Client: {e}")
    else:
        print("Warning: No API Key provided. Stage 2 & 3 will fail.")



###############
#Transcription#
###############

def transcribe_audio(audio_path, model_size="base", device="cpu", compute_type="int8"):
    """
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


def extract_words_from_whisper(whisper_result):
    words = []
    for segment in whisper_result.get("segments", []):
        if "words" in segment:
            for word_info in segment["words"]:
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0.0),
                    "end": word_info.get("end", 0.0),
                })
    return words


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


def calculate_wpm(words):
    if not words:
        return {"wpm": 0.0, "total_words": 0, "duration_seconds": 0.0}
    
    duration_seconds = words[-1]["end"] - words[0]["start"]
    wpm = (len(words) * 60.0) / duration_seconds if duration_seconds > 0 else 0.0
    
    return {
        "wpm": round(wpm, 2),
        "total_words": len(words),
        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_seconds / 60.0, 2),
    }


def detect_filler_words_per_sentence(sentences):
    single_word_fillers = ["um", "uh", "er", "ah", "hmm", "like", "so", "well", "actually", "basically", "right", "okay", "ok"]
    multi_word_fillers = ["you know", "i mean", "sort of", "kind of", "you see", "i guess", "i think"]
    
    filler_counts = []
    total_filler_words = 0

    for i, sentence in enumerate(sentences):
        count = 0
        sentence_lower = sentence.lower()
        for filler in single_word_fillers:
            count += len(re.findall(rf'\b{re.escape(filler)}\b', sentence_lower))
        for filler in multi_word_fillers:
            count += len(re.findall(rf'\b{re.escape(filler)}\b', sentence_lower))
            
        filler_counts.append({"sentence_index": i, "sentence": sentence, "filler_word_count": count})
        total_filler_words += count

    counts_only = [item["filler_word_count"] for item in filler_counts]
    summary = {
        "total_filler_words": total_filler_words,
        "total_sentences": len(sentences),
        "mean_filler_words_per_sentence": round(sum(counts_only) / len(counts_only), 2) if counts_only else 0.0,
        "sentences_with_filler_words": sum(1 for c in counts_only if c > 0),
    }

    return {"feature": "filler_word_analysis", "summary": summary}


# =============================
# ADVANCED FEATURES EXTRACTION=
# =============================

def check_snr(audio_path: str) -> float:
    audio = AudioSegment.from_file(audio_path).set_channels(1)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    samples /= float(2 ** (8 * audio.sample_width - 1))

    frame_len = int(sr * 0.02)  # 20ms frames
    frames = samples[: len(samples) // frame_len * frame_len].reshape(-1, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    noise_rms  = max(float(np.percentile(rms, 10)), 1e-12)
    signal_rms = max(float(np.median(rms[rms >= np.percentile(rms, 50)])), 1e-12)

    snr_db = 10.0 * np.log10(signal_rms ** 2 / noise_rms ** 2)

    return round(snr_db, 2)

def check_loudness(audio_path: str) -> float:
    """
    Calculates the integrated perceived loudness of an audio file in LUFS.
    """
    # 1. Load the audio and convert to mono, matching your SNR function
    audio = AudioSegment.from_file(audio_path).set_channels(1)
    sr = audio.frame_rate
    
    # 2. Extract samples and normalize them to float64 between -1.0 and 1.0
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    samples /= float(2 ** (8 * audio.sample_width - 1))
    
    # 3. Initialize the ITU-R BS.1770 meter using the sample rate
    meter = pyln.Meter(sr)
    
    # 4. Calculate the integrated loudness in LUFS
    loudness_lufs = meter.integrated_loudness(samples)
    
    return round(loudness_lufs, 2)

def get_pitch_expressiveness(audio_path, fmin=75, fmax=300):
    """
    Analyzes an audio file to determine pitch expressiveness.
    
    Parameters:
    - audio_path: Path to the audio file (e.g., .mp3, .wav)
    - fmin: Minimum expected pitch in Hz
    - fmax: Maximum expected pitch in Hz
    
    Returns:
    - float: The expressiveness score (Standard deviation of pitch in semitones).
    """
    # Load the audio file (supports mp3 directly)
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract F0 using pyin
    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    
    # Filter out unvoiced frames (silence/consonants)
    f0_voiced = f0[~np.isnan(f0)]
    
    if len(f0_voiced) == 0:
        raise ValueError("No voiced speech detected in the audio file.")
        
    # Convert Hz to Semitones
    f0_semitones = librosa.hz_to_midi(f0_voiced)
    
    # Calculate and return only the Standard Deviation
    return float(np.std(f0_semitones))

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


def run_advanced_pause_analysis(words, min_gap_sec=1.2):
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



def filter_contentful_sentences(sentences, min_tokens=5, filler_patterns=None):
    if filler_patterns is None:
        filler_patterns = ["um", "uh", "er", "ah", "hmm", "like", "so", "well", "actually", "basically", "right", "okay", "ok"]
    
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
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)

def smooth_similarities(similarities, window_size=3):
    if len(similarities) == 0: return similarities
    smoothed = []
    half_window = window_size // 2
    for i in range(len(similarities)):
        start, end = max(0, i - half_window), min(len(similarities), i + half_window + 1)
        smoothed.append(np.mean(similarities[start:end]))
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


def run_structure_evaluation_improved(sentences,
min_tokens=5,smoothing_window=3,depth_window=3,depth_alpha=0.8,hard_threshold=0.25,drift_threshold=0.4,min_intro_tokens=8,n_intro_sentences=3):
    """    
    Args:
        sentences: List of sentence strings
        min_tokens: Minimum tokens to keep a sentence
        smoothing_window: Window size for moving average smoothing
        depth_window: Window size for depth computation
        depth_alpha: Multiplier for std deviation in dynamic threshold (mean + alpha*std)
        hard_threshold: Hard fallback threshold for flow breaks (< this = definitely a break)
        drift_threshold: Threshold for drift detection (sim < this = off-topic)
        min_intro_tokens: Minimum tokens for intro sentence
        n_intro_sentences: Target number of intro sentences
    """

    model = SENTENCE_MODEL
    
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



##############
#Gemini agents
##############

def run_analysis_agent(feature_data, metrics_path, model_name="gemini-2.5-flash"):
    if GEMINI_CLIENT is None: return {"error": "Gemini client not initialized"}
    
    with open(metrics_path, "r") as f:
        metrics_thresholds = json.load(f)

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
        + json.dumps(feature_data, indent=2)
        + "\n\nMetrics Thresholds (metrics.json):\n"
        + json.dumps(metrics_thresholds, indent=2)
    )
    
    try:
        response = GEMINI_CLIENT.models.generate_content(model=model_name, contents=prompt)
        text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}



def run_feedback_agent(analysis_json, model_name="gemini-2.5-flash"):
    if GEMINI_CLIENT is None: return {"error": "Gemini client not initialized"}
    
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
        response = GEMINI_CLIENT.models.generate_content(model=model_name, contents=prompt)
        text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}