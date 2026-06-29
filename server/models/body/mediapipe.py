import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

# ---------------------------------------------------------------------------
# These constants are kept identical to openpose.py so every import site
# that does `from models.body.openpose import EMOTION_CLASSES, NUM_CLASSES`
# can be pointed here without any other change.
# ---------------------------------------------------------------------------

EMOTION_CLASSES = {
    0: "anger",
    1: "fear",
    2: "happiness",
    3: "neutral",
    4: "sadness",
}

NUM_CLASSES = len(EMOTION_CLASSES)

# Path to the MediaPipe Pose Landmarker .task model file.
# Download from:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
MEDIAPIPE_MODEL_PATH = r"D:\mediapipe_models\pose_landmarker_heavy.task"

# ---------------------------------------------------------------------------
# MediaPipe (33 landmarks) → H36M (17 joints) index mapping
#
# This is the exact semantic equivalent of the OpenPose BODY_25 → H36M
# mapping in preprocessing.py.  Every H36M joint maps to the same body
# part it did before; only the source index changes.
#
# MediaPipe 33-landmark indices used here:
#   0  NOSE
#   2  LEFT_EYE
#   5  RIGHT_EYE
#  11  LEFT_SHOULDER
#  12  RIGHT_SHOULDER
#  13  LEFT_ELBOW
#  14  RIGHT_ELBOW
#  15  LEFT_WRIST
#  16  RIGHT_WRIST
#  23  LEFT_HIP
#  24  RIGHT_HIP
#  25  LEFT_KNEE
#  26  RIGHT_KNEE
#  27  LEFT_ANKLE
#  28  RIGHT_ANKLE
#
# Synthesized joints (not directly present in MediaPipe):
#   MidHip → avg(LEFT_HIP=23,  RIGHT_HIP=24)   [was BODY_25 joint 8]
#   Neck   → avg(LEFT_SHOULDER=11, RIGHT_SHOULDER=12)  [was BODY_25 joint 1]
#
# H36M joint 7 (Spine) = avg(Neck, MidHip)
#   = avg( avg(11,12), avg(23,24) ) = avg(11, 12, 23, 24)
#   This is mathematically identical to the original avg(OP[1], OP[8]).
# ---------------------------------------------------------------------------

MEDIAPIPE_TO_H36M = {
    #  H36M idx : [MP landmark indices to average]
    0:  [23, 24],        # Hip center  ← avg(LEFT_HIP, RIGHT_HIP)       was OP[8]
    1:  [24],            # RHip        ← RIGHT_HIP                       was OP[9]
    2:  [26],            # RKnee       ← RIGHT_KNEE                      was OP[10]
    3:  [28],            # RAnkle      ← RIGHT_ANKLE                     was OP[11]
    4:  [23],            # LHip        ← LEFT_HIP                        was OP[12]
    5:  [25],            # LKnee       ← LEFT_KNEE                       was OP[13]
    6:  [27],            # LAnkle      ← LEFT_ANKLE                      was OP[14]
    7:  [11, 12, 23, 24],# Spine       ← avg(LS, RS, LH, RH)            was avg(OP[1],OP[8])
    8:  [11, 12],        # Thorax      ← avg(LEFT_SHOULDER,RIGHT_SHOULDER) was OP[1]
    9:  [0],             # Nose        ← NOSE                            was OP[0]
    10: [2, 5],          # Head        ← avg(LEFT_EYE, RIGHT_EYE)        was avg(OP[15],OP[16])
    11: [11],            # LShoulder   ← LEFT_SHOULDER                   was OP[5]
    12: [13],            # LElbow      ← LEFT_ELBOW                      was OP[6]
    13: [15],            # LWrist      ← LEFT_WRIST                      was OP[7]
    14: [12],            # RShoulder   ← RIGHT_SHOULDER                  was OP[2]
    15: [14],            # RElbow      ← RIGHT_ELBOW                     was OP[3]
    16: [16],            # RWrist      ← RIGHT_WRIST                     was OP[4]
}

# Total number of MediaPipe landmarks (fixed by the model)
_MP_NUM_LANDMARKS = 33


class MediaPipePose:
    """
    Drop-in replacement for the OpenPose class.

    Runs MediaPipe Pose Landmarker on frame images and returns keypoint arrays
    in the same (T, 33, 3) shape contract, where the third axis is
    (x, y, visibility) — directly analogous to OpenPose's (x, y, confidence).

    The public API mirrors OpenPose exactly:
        extract_keypoints(window)       → (T, 33, 3)
        extract_keypoints_batch(frames) → (N, 33, 3)
    """

    def __init__(
        self,
        model_path: str = MEDIAPIPE_MODEL_PATH,
        output_dir: str = "outputs/mediapipe_keypoints",
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Build a single re-usable landmarker in IMAGE mode so it can be
        # called frame-by-frame without re-initialising per call.
        base_options = BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame_path: str) -> np.ndarray:
        """
        Run the landmarker on a single image file.

        Returns:
            (33, 3) float32 array of (x, y, visibility).
            All zeros if no person is detected.
        """
        bgr = cv2.imread(frame_path)
        if bgr is None:
            return np.zeros((_MP_NUM_LANDMARKS, 3), dtype=np.float32)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return np.zeros((_MP_NUM_LANDMARKS, 3), dtype=np.float32)

        landmarks = result.pose_landmarks[0]   # first (and only) person
        kp = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )  # (33, 3)
        return kp

    # ------------------------------------------------------------------
    # Public API  (mirrors OpenPose)
    # ------------------------------------------------------------------

    def extract_keypoints(self, window: list) -> np.ndarray:
        """
        Process one window of frames.

        Args:
            window: list of image file paths

        Returns:
            np.ndarray of shape (T, 33, 3) — x, y, visibility per landmark
        """
        keypoints = [self._process_frame(fp) for fp in window]
        return np.stack(keypoints, axis=0)   # (T, 33, 3)

    def extract_keypoints_batch(self, frame_files: list) -> np.ndarray:
        """
        Process all frames in one pass (no subprocess overhead).

        Args:
            frame_files: all frames to process (already at target FPS)

        Returns:
            np.ndarray of shape (N, 33, 3) — one row per frame
        """
        keypoints = [self._process_frame(fp) for fp in frame_files]
        return np.stack(keypoints, axis=0)   # (N, 33, 3)

    def __del__(self):
        # Release the native landmarker handle cleanly.
        if hasattr(self, "_landmarker"):
            self._landmarker.close()