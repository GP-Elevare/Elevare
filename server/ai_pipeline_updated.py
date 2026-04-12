import sys
import json
import os
import cv2
import glob
import librosa
from pydub import AudioSegment

# from models.speech.speech_module import SpeechEmotionRecognizer
# from models.facial.facial_module import FacialEmotionRecognizer
from models.feedback.main_converted import run_full_pipeline
from models.body.body_module_but_better import BodyEmotionRecognizer
# from models.qg.T5 import generate_questions
# from models.qg.Text_Extractor import extract_text_from_pptx


# speech_model = SpeechEmotionRecognizer()
# facial_model = FacialEmotionRecognizer()
body_model   = BodyEmotionRecognizer()


def extract_audio(video_path, audio_path):
    """Extract audio from video using pydub."""
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")


# def feedback_module(video_path):
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video not found: {video_path}")

#     mp3_output_path = "outputs/audio.mp3"
#     audio = AudioSegment.from_file(video_path)
#     audio.export(mp3_output_path, format="mp3")

#     result = run_full_pipeline(
#         audio_path=mp3_output_path,
#         metrics_json_path="models/feedback/metrics.json"
#     )

#     with open("pipeline_output.json", "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=4)

#     print("Feedback pipeline finished. Output saved to pipeline_output.json")


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

    # # ===== AUDIO EXTRACTION =====
    # audio_path = "outputs/audio.wav"
    # extract_audio(video_path, audio_path)

    # # ===== AUDIO WINDOWING =====
    # window_seconds     = window_size / video_fps
    # y, sr              = librosa.load(audio_path, sr=None)
    # samples_per_window = int(window_seconds * sr)
    # audio_windows = [
    #     y[i:i + samples_per_window]
    #     for i in range(0, len(y), samples_per_window)
    #     if len(y[i:i + samples_per_window]) == samples_per_window
    # ]

    # ===== FRAME WINDOWING (facial) =====
    image_windows = [
        frame_files[i:i + window_size]
        for i in range(0, len(frame_files), window_size)
        if len(frame_files[i:i + window_size]) == window_size
    ]

    # ===== PREDICTIONS =====
    # speech_preds  = speech_model.predict(audio_windows, sr)
    # facial_preds  = facial_model.predict(image_windows)
    body_preds    = body_model.predict(frame_files, video_fps)   # uses new pipeline
    # feedback_module(video_path)

    return {"body": body_preds}


if __name__ == "__main__":
    video_path   = sys.argv[1]
    output_file  = sys.argv[2]
    window_size  = int(sys.argv[3]) if len(sys.argv) > 3 else 97

    results = process_video(video_path, window_size)

    base_path = output_file.replace('.json', '')

    # with open(f"{base_path}-speech-emotions.json", "w") as f:
    #     json.dump({"speech_emotions": results["speech"]}, f, indent=2)

    # with open(f"{base_path}-facial-emotions.json", "w") as f:
    #     json.dump({"facial_emotions": results["facial"]}, f, indent=2)

    with open(f"{base_path}-body-emotions.json", "w") as f:
        json.dump({"body_emotions": results.get("body", [])}, f, indent=2)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    # print(f"Speech emotions saved to {base_path}-speech-emotions.json")
    # print(f"Facial emotions saved to {base_path}-facial-emotions.json")
    print(f"Body emotions saved to {base_path}-body-emotions.json")