import math
import numpy as np
import torch


class Preprocessing:
    """
    Preprocesses raw OpenPose output for model prediction.

    Input  : (T, 25, 3) np.ndarray from OpenPose.extract_keypoints()
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
        0:  [8],       # Hip center  ← MidHip
        1:  [9],       # RHip
        2:  [10],      # RKnee
        3:  [11],      # RAnkle
        4:  [12],      # LHip
        5:  [13],      # LKnee
        6:  [14],      # LAnkle
        7:  [1, 8],    # Spine       ← avg(Neck, MidHip)
        8:  [1],       # Thorax      ← Neck
        9:  [0],       # Nose
        10: [15, 16],  # Head        ← avg(REye, LEye)
        11: [5],       # LShoulder
        12: [6],       # LElbow
        13: [7],       # LWrist
        14: [2],       # RShoulder
        15: [3],       # RElbow
        16: [4],       # RWrist
    }

    def __init__(self, target_len: int = 97, conf_threshold: float = 0.55):
        self.target_len     = target_len
        self.conf_threshold = conf_threshold

    def interpolate_missing(self, kps: np.ndarray) -> np.ndarray:
        """
        Interpolate low-confidence joints across time.
        Halves confidence of interpolated joints.

        Args:
            kps: (T, 25, 3)
        Returns:
            kps: (T, 25, 3) with gaps filled
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
        """
        Remap BODY_25 joints to H36M 17-joint layout.

        Args:
            kps: (T, 25, 3)
        Returns:
            out: (T, 17, 3)
        """
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

        Args:
            kps: (T, 17, 3)
        Returns:
            kps: (T, 17, 3) normalized
        """
        kps = kps.copy().astype(np.float32)
        kps[:, :, :2] -= kps[:, 0:1, :2].copy()

        torso_mean = np.linalg.norm(
            kps[:, 8, :2] - kps[:, 0, :2], axis=-1
        ).mean()

        if torso_mean > 1e-6:
            kps[:, :, :2] /= torso_mean

        return kps

    def pad_or_crop(self, kps: np.ndarray) -> np.ndarray:
        """
        Pad (repeat last frame) or uniformly subsample to target_len.

        Args:
            kps: (T, 17, 3)
        Returns:
            kps: (target_len, 17, 3)
        """
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
            kps: (T, 25, 3) raw OpenPose output from OpenPose.extract_keypoints()
        Returns:
            FloatTensor of shape (1, target_len, 17, 3), batched and ready for inference
        """
        kps = kps.copy().astype(np.float32)
        kps = self.interpolate_missing(kps)  # (T, 25, 3) — fill gaps
        kps = self.convert_to_h36m(kps)      # (T, 17, 3) — reindex joints
        kps = self.normalize_keypoints(kps)  # (T, 17, 3) — body-centered
        kps = self.pad_or_crop(kps)          # (target_len, 17, 3)

        return torch.FloatTensor(kps).unsqueeze(0)  # (1, target_len, 17, 3)


class FrameProcessor:
    """Handles FPS downsampling and window splitting for raw frame files."""

    def __init__(self, frame_files: list, source_fps: float, target_fps: float = 15):
        """
        Args:
            frame_files : sorted list of frame file paths
            source_fps  : FPS of the original video (cap.get(cv2.CAP_PROP_FPS))
            target_fps  : FPS to downsample to (default 15)
        """
        self.frame_files = frame_files
        self.source_fps  = source_fps
        self.target_fps  = target_fps

    def to_target_fps(self) -> list:
        """Subsample frame list from source_fps down to target_fps."""
        if self.source_fps <= self.target_fps:
            return self.frame_files

        step         = self.source_fps / self.target_fps
        total_frames = len(self.frame_files)
        indices      = [round(i * step) for i in range(math.floor(total_frames / step))]
        indices      = [i for i in indices if i < total_frames]

        return [self.frame_files[i] for i in indices]

    def split_into_windows(self, frames: list, window_size: int = 97) -> list:
        """Split a flat frame list into fixed-size windows, padding the last one if needed."""
        windows = []
        for i in range(0, len(frames), window_size):
            chunk = frames[i : i + window_size]
            if len(chunk) < window_size:
                chunk = self._pad_window(chunk, window_size)
            windows.append(chunk)
        return windows

    def _pad_window(self, frames: list, window_size: int) -> list:
        """Repeat the last frame until the window reaches window_size."""
        if not frames:
            raise ValueError("Cannot pad an empty frame list.")
        pad_count = window_size - len(frames)
        return frames + [frames[-1]] * pad_count