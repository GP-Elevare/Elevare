import subprocess
import os
import shutil
import glob
import json
import numpy as np
import torch
import pickle
from collections import Counter
import cv2
from scipy.interpolate import interp1d
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import json
import subprocess
import tempfile
import shutil
import math


output_json_dir = "outputs/openpose_keypoints"

# class openposing:
#     def extract_keypoints(window: list[str]) -> np.ndarray:
#     """
#     Extract OpenPose BODY_25 keypoints for each frame in a window.

#     Args:
#         window : list of frame file paths, len == window_size (e.g. 97)

#     Returns:
#         np.ndarray of shape (T, 25, 3)
#             T  = number of frames (== len(window))
#             25 = BODY_25 joints
#             3  = [x (pixel), y (pixel), confidence (0–1)]

#     Notes:
#         - If no person is detected in a frame, that frame's slice is zeros.
#         - If multiple people are detected, only the highest-confidence person is kept.
#         - Confidence == 0 means the joint was not detected; these will be
#           handled downstream by interpolate_missing().
#     """
#     raise NotImplementedError


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
 
    def forward(self, x):
        return self.net(x) + self.skip(x)
 
 
class MultiScaleTemporalCNN(nn.Module):
    def __init__(self, num_classes=5, num_joints=17, in_ch=2, hidden=256, dropout=0.5):
        super().__init__()
 
        joint_dim    = num_joints * in_ch
        bone_dim     = num_joints * 2
        velocity_dim = num_joints * 2
 
        def make_embed(in_dim, hidden, dropout):
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
 
        self.joint_embed    = make_embed(joint_dim,    hidden, dropout)
        self.bone_embed     = make_embed(bone_dim,     hidden, dropout)
        self.velocity_embed = make_embed(velocity_dim, hidden, dropout)
 
        def make_branches(h, drop):
            return nn.ModuleDict({
                'full': nn.Sequential(
                    TemporalConvBlock(h, h, kernel=3,  dropout=drop),
                    # TemporalConvBlock(h, h, kernel=3,  dropout=drop),
                ),
                'mid': nn.Sequential(
                    TemporalConvBlock(h, h, kernel=7,  dropout=drop),
                ),
                'long': nn.Sequential(
                    TemporalConvBlock(h, h, kernel=15, dropout=drop),
                ),
            })
 
        self.joint_branches    = make_branches(hidden, dropout)
        self.bone_branches     = make_branches(hidden, dropout)
        self.velocity_branches = make_branches(hidden, dropout)
 
        fused_dim = hidden * 3 * 3
        self.attn_pool = nn.Sequential(
            nn.Linear(fused_dim, 1),
            nn.Softmax(dim=1)
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
 
    def _compute_bones(self, x):
        parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        bones = x[:, :, :, :2].clone()
        for i, p in enumerate(parents):
            if i != p:
                bones[:, :, i, :] = x[:, :, i, :2] - x[:, :, p, :2]
            else:
                bones[:, :, i, :] = 0.0
        return bones
 
    def _compute_velocity(self, x):
        xy  = x[:, :, :, :2]
        vel = torch.zeros_like(xy)
        vel[:, 1:, :, :] = xy[:, 1:, :, :] - xy[:, :-1, :, :]
        return vel
 
    def _forward_stream(self, embed, branches, x_flat, B, T):
        x = embed(x_flat)
        x = x.reshape(B, T, -1).permute(0, 2, 1)
        f1 = branches['full'](x)
        f2 = branches['mid'](x)
        f3 = branches['long'](x)
        return torch.cat([f1, f2, f3], dim=1)
 
    def forward(self, x):
        x = x[..., :2]
        B, T, J, C = x.shape
 
        feat_joint = self._forward_stream(
            self.joint_embed, self.joint_branches,
            x.reshape(B*T, J*C), B, T
        )
        bones = self._compute_bones(x)
        feat_bone = self._forward_stream(
            self.bone_embed, self.bone_branches,
            bones.reshape(B*T, J*2), B, T
        )
        vel = self._compute_velocity(x)
        feat_vel = self._forward_stream(
            self.velocity_embed, self.velocity_branches,
            vel.reshape(B*T, J*2), B, T
        )
 
        fused  = torch.cat([feat_joint, feat_bone, feat_vel], dim=1)
        fused  = fused.permute(0, 2, 1)
        attn   = self.attn_pool(fused)
        pooled = (fused * attn).sum(dim=1)
        return self.head(pooled)

class Preprocessing:
    """
    Preprocesses raw OpenPose output for model prediction.

    Input  : (T, 25, 3) np.ndarray from extract_keypoints()
    Output : (1, T, 17, 3) FloatTensor ready for model inference
    """

    # ============================================================
    # OpenPose BODY_25 → H36M 17-joint mapping
    #  0=Hip(c), 1=RHip,  2=RKnee,   3=RAnkle
    #  4=LHip,   5=LKnee, 6=LAnkle,  7=Spine
    #  8=Thorax, 9=Nose,  10=Head,   11=LShoulder
    # 12=LElbow, 13=LWrist, 14=RShoulder, 15=RElbow, 16=RWrist
    # ============================================================

    OPENPOSE_TO_H36M = {
        0:  [8],      # Hip center  ← MidHip
        1:  [9],      # RHip
        2:  [10],     # RKnee
        3:  [11],     # RAnkle
        4:  [12],     # LHip
        5:  [13],     # LKnee
        6:  [14],     # LAnkle
        7:  [1, 8],   # Spine       ← avg(Neck, MidHip)
        8:  [1],      # Thorax      ← Neck
        9:  [0],      # Nose
        10: [15, 16], # Head        ← avg(REye, LEye)
        11: [5],      # LShoulder
        12: [6],      # LElbow
        13: [7],      # LWrist
        14: [2],      # RShoulder
        15: [3],      # RElbow
        16: [4],      # RWrist
    }

    def __init__(self, target_len: int = 97, conf_threshold: float = 0.55):
        self.target_len     = target_len
        self.conf_threshold = conf_threshold

    def interpolate_missing(self, kps: np.ndarray) -> np.ndarray:
        """
        Interpolate low-confidence joints across time. Halves confidence of interpolated joints.
        kps: (T, 25, 3)
        """
        coords = kps[:, :, :2].copy()
        conf   = kps[:, :, 2].copy()

        T, V, _ = coords.shape
        t = np.arange(T)

        for v in range(V):
            valid = conf[:, v] >= self.conf_threshold
            if valid.sum() < 2:
                continue
            for c in range(2):
                coords[:, v, c] = np.interp(t, t[valid], coords[valid, v, c])
            conf[~valid, v] *= 0.5

        kps[:, :, :2] = coords
        kps[:, :, 2]  = conf
        return kps

    def convert_to_h36m(self, kps: np.ndarray) -> np.ndarray:
        """(T, 25, 3) → (T, 17, 3)"""
        T   = kps.shape[0]
        out = np.zeros((T, 17, 3), dtype=np.float32)
        for h36m_idx, op_idxs in self.OPENPOSE_TO_H36M.items():
            out[:, h36m_idx, :] = kps[:, op_idxs, :].mean(axis=1)
        return out

    def normalize_keypoints(self, kps: np.ndarray) -> np.ndarray:
        """
        Body-centered normalization:
          1. Subtract hip center (joint 0) — removes camera position dependency.
          2. Scale by mean torso height (hip→thorax) — removes distance-to-camera dependency.
        kps: (T, 17, 3)
        """
        kps = kps.copy().astype(np.float32)

        kps[:, :, :2] -= kps[:, 0:1, :2].copy()   # center on hip

        torso_mean = np.linalg.norm(
            kps[:, 8, :2] - kps[:, 0, :2], axis=-1
        ).mean()

        if torso_mean > 1e-6:
            kps[:, :, :2] /= torso_mean

        return kps

    def pad_or_crop(self, kps: np.ndarray) -> np.ndarray:
        """Pad (repeat last frame) or uniformly subsample to target_len."""
        T = len(kps)
        if T == self.target_len:
            return kps
        elif T < self.target_len:
            pad = np.tile(kps[-1:], (self.target_len - T, 1, 1))
            return np.concatenate([kps, pad], axis=0)
        else:
            idx = np.linspace(0, T - 1, self.target_len, dtype=int)
            return kps[idx]

    def process_window(self, kps: np.ndarray) -> torch.FloatTensor:
        """
        Full preprocessing pipeline for one window.

        Args:
            kps : (T, 25, 3) raw OpenPose output from extract_keypoints()

        Returns:
            FloatTensor of shape (1, target_len, 17, 3)
            — batched and ready for model.predict() / model.forward()
        """
        kps = kps.copy().astype(np.float32)

        kps = self.interpolate_missing(kps)   # (T, 25, 3)  fill gaps
        kps = self.convert_to_h36m(kps)       # (T, 17, 3)  reindex joints
        kps = self.normalize_keypoints(kps)   # (T, 17, 3)  body-centered
        kps = self.pad_or_crop(kps)           # (target_len, 17, 3)

        return torch.FloatTensor(kps).unsqueeze(0)  # (1, target_len, 17, 3)

class FrameProcessor:
    def __init__(self, frame_files, source_fps, target_fps = 15):
        """
        Args:
            frame_files : sorted list of frame file paths (from extraction step)
            source_fps  : FPS of the original video (cap.get(cv2.CAP_PROP_FPS))
            target_fps  : FPS to downsample to (default 15)
        """
        self.frame_files = frame_files
        self.source_fps  = source_fps
        self.target_fps  = target_fps

    def to_target_fps(self):
        if self.source_fps <= self.target_fps:
            return self.frame_files

        step          = self.source_fps / self.target_fps
        total_frames  = len(self.frame_files)
        indices       = [round(i * step) for i in range(math.floor(total_frames / step))]
        indices       = [i for i in indices if i < total_frames]

        return [self.frame_files[i] for i in indices]

    def split_into_windows(self, frames, window_size = 97):
        windows = []
        for i in range(0, len(frames), window_size):
            chunk = frames[i:i + window_size]
            if len(chunk) < window_size:
                chunk = self.pad_window(chunk, window_size)
            windows.append(chunk)
        return windows

    def pad_window(self, frames, window_size):
        if not frames:
            raise ValueError("Cannot pad an empty frame list.")
        pad_count = window_size - len(frames)
        return frames + [frames[-1]] * pad_count

EMOTION_CLASSES = {
    0: 'anger',
    1: 'fear',
    2: 'happiness',
    3: 'neutral',
    4: 'sadness',
}

NUM_CLASSES = len(EMOTION_CLASSES)

OPENPOSE_DIR = "D:/openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d/openpose"


import os
import glob
import json
import subprocess
import tempfile
import shutil
import numpy as np


OPENPOSE_DIR = "D:\openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d\openpose"   

class OpenPose:
    def __init__(self, openpose_dir: str = OPENPOSE_DIR, output_dir: str = "outputs/openpose_keypoints"):
        self.openpose_dir = openpose_dir
        self.output_dir   = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Windows CPU build ships the demo exe directly in bin/
        self.binary = os.path.join(openpose_dir, "bin", "OpenPoseDemo.exe")

    def _run_openpose(self, input_dir, json_output_dir):
        cmd = [
            self.binary,
            "--image_dir",         input_dir,
            "--write_json",        json_output_dir,
            "--display",           "0",
            "--render_pose",       "0",
            "--model_pose",        "BODY_25",
            "--number_people_max", "1",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.openpose_dir   # ← must run from OpenPose root
        )
        if result.returncode != 0:
            raise RuntimeError(f"OpenPose failed:\n{result.stderr}")

    def _load_json(self, json_path: str) -> np.ndarray:
        """
        Parse one OpenPose JSON file.
        Returns (25, 3) array — zeros if no person detected.
        """
        with open(json_path) as f:
            data = json.load(f)

        people = data.get("people", [])
        if not people:
            return np.zeros((25, 3), dtype=np.float32)

        flat = np.array(people[0]["pose_keypoints_2d"], dtype=np.float32)
        return flat.reshape(25, 3)

    def extract_keypoints(self, window) -> np.ndarray:
        # 1. Copy window frames into a temp input folder
        tmp_input = tempfile.mkdtemp(prefix="op_input_")
        tmp_json  = tempfile.mkdtemp(prefix="op_json_")

        try:
            for i, frame_path in enumerate(window):
                ext = os.path.splitext(frame_path)[1]
                dst = os.path.join(tmp_input, f"frame_{i:05d}{ext}")
                shutil.copy(frame_path, dst)

            # 2. Run OpenPose on the temp input folder
            self._run_openpose(tmp_input, tmp_json)

            # 3. Parse JSONs in order
            json_files = sorted(glob.glob(os.path.join(tmp_json, "*.json")))
            keypoints  = []

            for i in range(len(window)):
                # OpenPose names output as <input_name>_keypoints.json
                expected = os.path.join(tmp_json, f"frame_{i:05d}_keypoints.json")
                if os.path.exists(expected):
                    kp = self._load_json(expected)
                elif json_files:
                    # fallback: take by position if naming differs
                    kp = self._load_json(json_files[i]) if i < len(json_files) else np.zeros((25, 3), dtype=np.float32)
                else:
                    kp = np.zeros((25, 3), dtype=np.float32)
                keypoints.append(kp)

            # 4. Persist JSONs to the permanent output dir (one subdir per call)
            window_id  = os.path.basename(tmp_input)
            output_sub = os.path.join(self.output_dir, window_id)
            shutil.copytree(tmp_json, output_sub)

        finally:
            shutil.rmtree(tmp_input, ignore_errors=True)
            shutil.rmtree(tmp_json,  ignore_errors=True)

        return np.stack(keypoints, axis=0)   # (T, 25, 3)

class BodyEmotionRecognizer:

    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_modeling.pth')

        self.model = MultiScaleTemporalCNN(
            num_classes=NUM_CLASSES,
            num_joints=17,
            in_ch=2,
            hidden=128,
            dropout=0.5,
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_window(self, keypoints: torch.FloatTensor) -> str:
        keypoints = keypoints.to(self.device)

        with torch.no_grad():
            logits = self.model(keypoints)          # (1, 5)
            pred   = logits.argmax(dim=1).item()

        return EMOTION_CLASSES[pred]

    def predict(self, frame_files, source_fps):
        """
        Full inference pipeline from raw frame files to emotion labels.

        Args:
            frame_files : sorted list of frame file paths
            source_fps  : FPS of the source video

        Returns:
            List of emotion label strings, one per window
        """
        frame_processor = FrameProcessor(frame_files, source_fps)
        preprocessor    = Preprocessing()
        openpose        = OpenPose()

        frames_15fps = frame_processor.to_target_fps()
        windows      = frame_processor.split_into_windows(frames_15fps)

        predictions = []
        for window in windows:
            keypoints = openpose.extract_keypoints(window)   # (T, 25, 3)
            keypoints = preprocessor.process_window(keypoints)  # (1, 97, 17, 3)
            label     = self.predict_window(keypoints)
            predictions.append(label)

        return predictions

