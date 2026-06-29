import sys
import json
from pathlib import Path
import shutil
from models.speech.speech_module import SpeechEmotionRecognizer
from models.facial.facial_module import FacialEmotionRecognizer
from models.feedback.main_converted import run_full_pipeline
from models.feedback import eye_gaze
from models.feedback import feedback_engine
from models.body.body_module import BodyEmotionRecognizer
import os, cv2, glob, librosa, io
from pydub import AudioSegment
from io import BytesIO

speech_model = SpeechEmotionRecognizer()
facial_model = FacialEmotionRecognizer()
body_model = BodyEmotionRecognizer()

#Test
API_KEY = "AIzaSyDTwm2LsemdKYOS9-68GbearQgZDZEaNjQ" 

def extract_audio(video_path, audio_path):
    """Extract audio from video using pydub."""
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")

def feedback_module(video_path, fps=5):
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    mp3_output_path = "outputs/audio.mp3"
    audio = AudioSegment.from_file(video_path)
    audio.export(mp3_output_path, format="mp3")

    METRICS_JSON_PATH = "models/feedback/metrics.json"
    OUTPUT_PATH = "pipeline_output.json"

    result = run_full_pipeline(
        audio_path=mp3_output_path,
        metrics_json_path=METRICS_JSON_PATH
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("Pipeline finished successfully!")
    print("Output saved to:", OUTPUT_PATH)

def new_feedback_module(video_path):
    print("--- Initializing Feedback Engine ---\n")
    feedback_engine.initialize_models(api_key=API_KEY)
    mp3_output_path = "outputs/audio.mp3"
    audio = AudioSegment.from_file(video_path)
    audio.export(mp3_output_path, format="mp3")
    METRICS_PATH = "models/feedback/metrics.json"
    
    print("--- Stage 1: Transcription ---\n")
    whisper_result = feedback_engine.transcribe_audio(mp3_output_path)
    words = feedback_engine.extract_words_from_whisper(whisper_result)
    sentences = feedback_engine.extract_sentences_from_whisper(whisper_result)
    print(f"Transcribed {len(words)} words and {len(sentences)} sentences.\n")

    print("--- Stage 2: Feature Extraction ---\n")
    wpm_result = feedback_engine.calculate_wpm(words)
    filler_result = feedback_engine.detect_filler_words_per_sentence(sentences)
    pause_result = feedback_engine.run_advanced_pause_analysis(words)
    structure_result = feedback_engine.run_structure_evaluation_improved(sentences,min_tokens=5,smoothing_window=3,depth_window=3,depth_alpha=0.8,
    hard_threshold=0.25,drift_threshold=0.4,min_intro_tokens=8,n_intro_sentences=3)
    #Audio Frame Analysis goz2 malek
    snr_result = feedback_engine.check_snr(mp3_output_path)
    pitch_result = feedback_engine.get_pitch_expressiveness(mp3_output_path)
    loudness_result = feedback_engine.check_loudness(mp3_output_path)
    #feedback_engine.process_audio_per_frame(AUDIO_PATH, "frame_stats.json")
    gaze_result = eye_gaze.analyze_video_gaze_headless(video_path)
    # Combine all data for the AI Agent
    raw_data_json = {
        "words_per_minute": wpm_result,
        "filler_word_analysis": filler_result,
        "advanced_pause_analysis": pause_result,
        "structure_evaluation": structure_result,
        "gaze_analysis": gaze_result,
        "snr_db": snr_result,
        "pitch_variance_hz": pitch_result,
        "loudness_db": loudness_result
    }
    output_filename = "presentation_feedback_raw.json"
    
    with open(output_filename, "w", encoding="utf-8") as json_file:
        # indent=4 makes the file easily readable instead of one giant line
        json.dump(raw_data_json, json_file, indent=4)
        
    print(f"Raw data successfully saved to {output_filename}")

    print("--- Stage 3: AI Analysis & Feedback ---\n")
    # Run Analysis Agent
    analysis_output = feedback_engine.run_analysis_agent(raw_data_json, METRICS_PATH)
    # Run Feedback Agent
    feedback_output = feedback_engine.run_feedback_agent(analysis_output)

    print("=== FINAL FEEDBACK ===\n")
    with open("final_feedback.json", "w") as f:
        json.dump(feedback_output, f, indent=2)
    with open("analysis_feedback.json", "w") as f:
        json.dump(analysis_output, f, indent=2)
    print("Feedback saved to final_feedback.json")



def extract_frames(video_path, frames_dir):
    """Extract all frames from video. Returns (frame_files, video_fps)."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{frames_dir}/frame_{frame_count:05d}.jpg", frame)
        frame_count += 1
    cap.release()
    frame_files = sorted(glob.glob(f"{frames_dir}/*.jpg"))
    return frame_files, video_fps    

def process_video(video_path, window_size=5):
    # ===== FRAME EXTRACTION =====
    frame_files, video_fps = extract_frames(video_path, "outputs/frames")

    # ===== AUDIO EXTRACTION =====
    audio_path = "outputs/audio.wav"
    extract_audio(video_path, audio_path)

    # ===== AUDIO WINDOWING =====
    window_seconds     = window_size / video_fps
    y, sr              = librosa.load(audio_path, sr=None)
    samples_per_window = int(window_seconds * sr)
    audio_windows = [
        y[i:i + samples_per_window]
        for i in range(0, len(y), samples_per_window)
        if len(y[i:i + samples_per_window]) == samples_per_window
    ]

    # ===== FRAME WINDOWING (facial) =====
    image_windows = [
        frame_files[i:i + window_size]
        for i in range(0, len(frame_files), window_size)
        if len(frame_files[i:i + window_size]) == window_size
    ]

    # ===== PREDICTIONS =====
    speech_preds  = speech_model.predict(audio_windows, sr)
    facial_preds  = facial_model.predict(image_windows)
    body_preds    = body_model.predict(frame_files, video_fps)   
    new_feedback_module(video_path)

    return {"speech": speech_preds, "facial": facial_preds, "body": body_preds}
    
if __name__ == "__main__":
    video_path = sys.argv[1]  # get video path from Node
    output_file = sys.argv[2]  # base path to save JSON results
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 97  # fps = frames per window, default 5
    
    # Create separate files for speech, facial and body emotions
    results = process_video(video_path, window_size)
    base_path = output_file.replace('.json', '')
    
    speech_output = f"{base_path}-speech-emotions.json"
    facial_output = f"{base_path}-facial-emotions.json"
    body_output = f"{base_path}-body-emotions.json"
    
    # Save speech results
    with open(speech_output, "w") as f:
        json.dump({"speech_emotions": results["speech"]}, f, indent=2)
    
    # Save facial results 
    with open(facial_output, "w") as f:
        json.dump({"facial_emotions": results["facial"]}, f, indent=2)
    
    # Save body results 
    with open(body_output, "w") as f:
        json.dump({"body_emotions": results.get("body", [])}, f, indent=2)
    
    # Save combined results to main file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    with open("final_emotions.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Speech emotions saved to {speech_output}")
    print(f"Facial emotions saved to {facial_output}")
    print(f"Body emotions saved to {body_output}")
