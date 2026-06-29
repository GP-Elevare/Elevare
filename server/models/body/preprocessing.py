import math
import numpy as np
import torch
from multiprocessing import Pool, cpu_count

from models.body.mediapipe import MEDIAPIPE_TO_H36M

# Number of landmarks the MediaPipe model outputs
_MP_NUM_LANDMARKS = 33


class Preprocessing:
    """
    Preprocesses raw MediaPipe Pose output for model prediction.

    Input  : (T, 33, 3) np.ndarray from MediaPipePose.extract_keypoints()
    Output : (1, T, 17, 3) FloatTensor ready for model inference

    All logic (interpolation, H36M conversion, normalization, padding) is
    identical to the OpenPose version.  Only the source landmark indices
    in the joint mapping have changed to reflect MediaPipe's 33-landmark
    layout — every H36M joint still maps to the same body part.
    """

    # MEDIAPIPE_TO_H36M is imported from mediapipe_pose.py and referenced
    # here to keep the mapping in one place.
    MEDIAPIPE_TO_H36M = MEDIAPIPE_TO_H36M

    def __init__(self, target_len: int = 97, conf_threshold: float = 0.55):
        self.target_len     = target_len
        self.conf_threshold = conf_threshold

    def interpolate_missing(self, kps: np.ndarray) -> np.ndarray:
        """
        Interpolate low-confidence joints across time.
        Halves confidence of interpolated joints.

        Args:
            kps: (T, 33, 3)
        Returns:
            kps: (T, 33, 3) with gaps filled
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
        Remap MediaPipe 33-landmark layout to H36M 17-joint layout.

        Each H36M joint is the mean of one or more MediaPipe landmarks that
        correspond to the same anatomical point.  This is semantically
        identical to the original OpenPose -> H36M conversion.

        Args:
            kps: (T, 33, 3)
        Returns:
            out: (T, 17, 3)
        """
        T   = kps.shape[0]
        out = np.zeros((T, 17, 3), dtype=np.float32)
        for h36m_idx, mp_idxs in self.MEDIAPIPE_TO_H36M.items():
            out[:, h36m_idx, :] = kps[:, mp_idxs, :].mean(axis=1)
        return out

    def normalize_keypoints(self, kps: np.ndarray) -> np.ndarray:
        """
        Body-centered normalization:
          1. Subtract hip center (joint 0) -- removes camera position dependency.
          2. Scale by mean torso height (hip->thorax) -- removes distance-to-camera dependency.

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
            kps: (T, 33, 3) raw MediaPipe output
        Returns:
            FloatTensor of shape (1, target_len, 17, 3)
        """
        kps = kps.copy().astype(np.float32)
        kps = self.interpolate_missing(kps)  # (T, 33, 3) -- fill gaps
        kps = self.convert_to_h36m(kps)      # (T, 17, 3) -- reindex joints
        kps = self.normalize_keypoints(kps)  # (T, 17, 3) -- body-centered
        kps = self.pad_or_crop(kps)          # (target_len, 17, 3)

        return torch.FloatTensor(kps).unsqueeze(0)  # (1, target_len, 17, 3)

    def process_windows_parallel(
        self, windows: list, n_workers: int = None
    ) -> torch.FloatTensor:
        """
        Option 5: preprocess all windows in parallel across CPU cores.

        process_window is pure NumPy with no shared state, making it
        embarrassingly parallel. Uses a process pool so the GIL is not
        a bottleneck.

        Args:
            windows  : list of (T, 33, 3) arrays -- one per window
            n_workers: number of worker processes; defaults to cpu_count()

        Returns:
            FloatTensor of shape (B, target_len, 17, 3) -- all windows stacked
        """
        n_workers = n_workers or cpu_count()

        with Pool(processes=n_workers) as pool:
            tensors = pool.map(self.process_window, windows)

        return torch.cat(tensors, dim=0)  # (B, target_len, 17, 3)


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