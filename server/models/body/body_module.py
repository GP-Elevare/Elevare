import os
import numpy as np
import torch

from models.body.models import MultiScaleTemporalCNN, TemporalConvBlock
from models.body.mediapipe import MediaPipePose, EMOTION_CLASSES, NUM_CLASSES
from models.body.preprocessing import FrameProcessor, Preprocessing


class BodyEmotionRecognizer:

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_modeling.pth")

        self.model = MultiScaleTemporalCNN(
            num_classes=NUM_CLASSES,
            num_joints=17,
            in_ch=2,
            hidden=128,
            dropout=0.5,
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Option 4: fuse kernels and optimise the compute graph.
        # Adds a one-time compilation cost on the first forward call,
        # then gives 20-40% faster inference on every subsequent call.
        self.model = torch.compile(self.model)

    def predict_window(self, keypoints: torch.FloatTensor) -> str:
        """
        Run inference on a single preprocessed window.

        Args:
            keypoints: FloatTensor of shape (1, T, 17, 3)

        Returns:
            Predicted emotion label string
        """
        keypoints = keypoints.to(self.device)
        with torch.no_grad():
            logits = self.model(keypoints)
            pred   = logits.argmax(dim=1).item()
        return EMOTION_CLASSES[pred]

    def predict(self, frame_files: list, source_fps: float) -> list:
        """
        Full inference pipeline from raw frame files to per-window emotion labels.

        Args:
            frame_files : sorted list of frame file paths
            source_fps  : FPS of the source video

        Returns:
            List of emotion label strings, one per window
        """
        frame_processor = FrameProcessor(frame_files, source_fps)
        preprocessor    = Preprocessing()
        pose            = MediaPipePose()

        # Downsample FPS
        frames_15fps = frame_processor.to_target_fps()

        # Option 1: extract keypoints for ALL frames in a single pass
        # (no subprocess overhead -- MediaPipe runs in-process).
        all_keypoints = pose.extract_keypoints_batch(frames_15fps)  # (N, 33, 3)

        # Split the keypoints array into 97-frame windows (mirrors
        # split_into_windows but operates directly on the numpy array).
        window_size = preprocessor.target_len
        n_frames    = len(all_keypoints)
        windows_kp  = []

        for start in range(0, n_frames, window_size):
            chunk = all_keypoints[start : start + window_size]
            if len(chunk) < window_size:
                pad   = np.tile(chunk[-1:], (window_size - len(chunk), 1, 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            windows_kp.append(chunk)

        # Option 5: preprocess all windows in parallel across CPU cores,
        # then run a single batched forward pass on the GPU.
        batch = preprocessor.process_windows_parallel(windows_kp)  # (B, 97, 17, 3)

        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(batch)              # (B, num_classes)
            preds  = logits.argmax(dim=1).tolist()

        return [EMOTION_CLASSES[p] for p in preds]