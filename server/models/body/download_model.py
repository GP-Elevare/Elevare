# import urllib.request
# import os

# url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
# output_path = "models/body/pose_landmarker_full.task"

# os.makedirs(os.path.dirname(output_path), exist_ok=True)

# print(f"Downloading model from {url}...")
# try:
#     urllib.request.urlretrieve(url, output_path)
#     print(f"Model downloaded to {output_path}")
# except Exception as e:
#     print(f"Download failed: {e}")
#     exit(1)
