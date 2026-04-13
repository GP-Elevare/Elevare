import os
import torch


from models.body.models import MultiScaleTemporalCNN, TemporalConvBlock
from models.body.openpose import OpenPose, EMOTION_CLASSES, NUM_CLASSES, OPENPOSE_DIR
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
            logits = self.model(keypoints)       # (1, num_classes)
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
        openpose        = OpenPose()

        frames_15fps = frame_processor.to_target_fps()
        windows      = frame_processor.split_into_windows(frames_15fps)

        predictions = []
        for window in windows:
            keypoints = openpose.extract_keypoints(window)      # (T, 25, 3)
            keypoints = preprocessor.process_window(keypoints)  # (1, 97, 17, 3)
            label     = self.predict_window(keypoints)
            predictions.append(label)

        return predictions