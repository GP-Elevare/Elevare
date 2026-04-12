import sys
import json
from pathlib import Path
import shutil
from models.speech.speech_module import SpeechEmotionRecognizer
from models.facial.facial_module import FacialEmotionRecognizer
from models.feedback.main_converted import run_full_pipeline
from models.body.body_module import MediaPipeExtractor, BodyEmotionRecognizer, output_json_dir
import os, cv2, glob, librosa, io
from pydub import AudioSegment
from io import BytesIO

speech_model = SpeechEmotionRecognizer()
facial_model = FacialEmotionRecognizer()
body_model = BodyEmotionRecognizer()
extractor = MediaPipeExtractor()

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

def process_video(video_path, window_size = 5):
    os.makedirs("outputs/frames", exist_ok = True)
    
    # Openpose Keypoint Extraction 
    extractor.extract_from_video(video_path, output_json_dir)   

    # ===== AUDIO EXTRACTION =====
    audio_path = "outputs/audio.wav"
    extract_audio(video_path, audio_path)
    
    # ===== FRAME EXTRACTION =====
    frames_dir = "outputs/frames"
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)  # Get actual video FPS
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{frames_dir}/frame_{frame_count:05d}.jpg", frame)
        frame_count += 1
    cap.release()
    
    # fps parameter = frames per window
    # Calculate seconds per window from frames
    window_seconds = window_size / video_fps  # frames ÷ (frames/sec) = seconds
    
    # ===== WINDOWING audio and frames =====
    y, sr = librosa.load(audio_path, sr=None)
    samples_per_window = int(window_seconds * sr)  # seconds * sample rate = samples per window
    audio_windows = [
        y[i:i + samples_per_window]
        for i in range(0, len(y), samples_per_window)
        if len(y[i:i + samples_per_window]) == samples_per_window
    ]
    
    # Use fps directly as frames per window
    frame_files = sorted(glob.glob(f"{frames_dir}/*.jpg"))
    image_windows = [
        frame_files[i:i + window_size]
        for i in range(0, len(frame_files), window_size)
        if len(frame_files[i:i + window_size]) == window_size
    ]
    
    # ===== PREDICTIONS =====
    speech_preds = speech_model.predict(audio_windows, sr)
    facial_preds = facial_model.predict(image_windows)
    body_preds = body_model.predict(image_windows)
    feedback_module(video_path)

    return {"speech": speech_preds, "facial": facial_preds, "body": body_preds}
    
if __name__ == "__main__":
    video_path = sys.argv[1]  # get video path from Node
    output_file = sys.argv[2]  # base path to save JSON results
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 97  # fps = frames per window, default 5
    
    results = process_video(video_path, window_size)
    
    # Create separate files for speech, facial and body emotions
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
    
    print(f"Results saved to {output_file}")
    print(f"Speech emotions saved to {speech_output}")
    print(f"Facial emotions saved to {facial_output}")
    print(f"Body emotions saved to {body_output}")
