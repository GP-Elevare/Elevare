# import os
# import glob
# import json
# import shutil
# import subprocess
# import tempfile
# import numpy as np

# OPENPOSE_DIR = r"D:\openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d\openpose"

# EMOTION_CLASSES = {
#     0: "anger",
#     1: "fear",
#     2: "happiness",
#     3: "neutral",
#     4: "sadness",
# }

# NUM_CLASSES = len(EMOTION_CLASSES)


# class OpenPose:
#     """Thin wrapper around the OpenPose CLI for extracting 2-D body keypoints."""

#     def __init__(
#         self,
#         openpose_dir: str = OPENPOSE_DIR,
#         output_dir: str = "outputs/openpose_keypoints",
#     ):
#         self.openpose_dir = openpose_dir
#         self.output_dir   = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)

#         self.binary = os.path.join(openpose_dir, "bin", "OpenPoseDemo.exe")

#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------

#     def _run_openpose(self, input_dir: str, json_output_dir: str) -> None:
#         cmd = [
#             self.binary,
#             "--image_dir",         input_dir,
#             "--write_json",        json_output_dir,
#             "--display",           "0",
#             "--render_pose",       "0",
#             "--model_pose",        "BODY_25",
#             "--number_people_max", "1",
#             "--net_resolution",    "320x176",
#         ]
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             cwd=self.openpose_dir,
#         )
#         if result.returncode != 0:
#             raise RuntimeError(f"OpenPose failed:\n{result.stderr}")

#     def _load_json(self, json_path: str) -> np.ndarray:
#         """
#         Parse one OpenPose JSON file.

#         Returns:
#             (25, 3) float32 array — zeros if no person detected.
#         """
#         with open(json_path) as f:
#             data = json.load(f)

#         people = data.get("people", [])
#         if not people:
#             return np.zeros((25, 3), dtype=np.float32)

#         flat = np.array(people[0]["pose_keypoints_2d"], dtype=np.float32)
#         return flat.reshape(25, 3)

#     def _link_or_copy(self, src: str, dst: str) -> None:
#         """
#         Option 2: prefer a symlink (near-zero cost); fall back to copy when
#         src and dst are on different filesystems (e.g. network drive vs temp).
#         """
#         try:
#             os.symlink(os.path.abspath(src), dst)
#         except (OSError, NotImplementedError):
#             shutil.copy(src, dst)

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def extract_keypoints(self, window: list) -> np.ndarray:
#         """
#         Run OpenPose on a list of frame paths and return stacked keypoints.

#         Args:
#             window: list of image file paths (one window of frames)

#         Returns:
#             np.ndarray of shape (T, 25, 3) — x, y, confidence per joint
#         """
#         tmp_input = tempfile.mkdtemp(prefix="op_input_")
#         tmp_json  = tempfile.mkdtemp(prefix="op_json_")

#         try:
#             for i, frame_path in enumerate(window):
#                 ext = os.path.splitext(frame_path)[1]
#                 dst = os.path.join(tmp_input, f"frame_{i:05d}{ext}")
#                 self._link_or_copy(frame_path, dst)  # Option 2

#             self._run_openpose(tmp_input, tmp_json)

#             json_files = sorted(glob.glob(os.path.join(tmp_json, "*.json")))
#             keypoints  = []

#             for i in range(len(window)):
#                 expected = os.path.join(tmp_json, f"frame_{i:05d}_keypoints.json")
#                 if os.path.exists(expected):
#                     kp = self._load_json(expected)
#                 elif i < len(json_files):
#                     kp = self._load_json(json_files[i])
#                 else:
#                     kp = np.zeros((25, 3), dtype=np.float32)
#                 keypoints.append(kp)

#             window_id  = os.path.basename(tmp_input)
#             output_sub = os.path.join(self.output_dir, window_id)
#             shutil.copytree(tmp_json, output_sub)

#         finally:
#             shutil.rmtree(tmp_input, ignore_errors=True)
#             shutil.rmtree(tmp_json,  ignore_errors=True)

#         return np.stack(keypoints, axis=0)  # (T, 25, 3)

#     def extract_keypoints_batch(self, frame_files: list) -> np.ndarray:
#         """
#         Option 1: run OpenPose ONCE over all frames instead of once per window.
#         Eliminates repeated binary cold-starts and temp-dir overhead.

#         Args:
#             frame_files: all frames to process (already at target FPS)

#         Returns:
#             np.ndarray of shape (N, 25, 3) — one row per frame
#         """
#         tmp_input = tempfile.mkdtemp(prefix="op_input_")
#         tmp_json  = tempfile.mkdtemp(prefix="op_json_")

#         try:
#             for i, fp in enumerate(frame_files):
#                 ext = os.path.splitext(fp)[1]
#                 dst = os.path.join(tmp_input, f"frame_{i:06d}{ext}")
#                 self._link_or_copy(fp, dst)

#             self._run_openpose(tmp_input, tmp_json)

#             json_files = sorted(glob.glob(os.path.join(tmp_json, "*.json")))
#             keypoints  = []

#             for i in range(len(frame_files)):
#                 expected = os.path.join(tmp_json, f"frame_{i:06d}_keypoints.json")
#                 if os.path.exists(expected):
#                     kp = self._load_json(expected)
#                 elif i < len(json_files):
#                     kp = self._load_json(json_files[i])
#                 else:
#                     kp = np.zeros((25, 3), dtype=np.float32)
#                 keypoints.append(kp)

#             window_id  = os.path.basename(tmp_input)
#             output_sub = os.path.join(self.output_dir, window_id)
#             shutil.copytree(tmp_json, output_sub)

#         finally:
#             shutil.rmtree(tmp_input, ignore_errors=True)
#             shutil.rmtree(tmp_json,  ignore_errors=True)

#         return np.stack(keypoints, axis=0)  # (N, 25, 3)