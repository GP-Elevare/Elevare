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


try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print(f"Warning: MediaPipe not available - {e}")

class MediaPipeExtractor:
    def __init__(self):
        """Initialize Pose detection using MediaPipe Tasks API"""
        self.pose_landmarker = None
        self.backend = "mediapipe_tasks"
        
        if not MEDIAPIPE_AVAILABLE:
            print("MediaPipe not available, using fallback mode")
            self.backend = "fallback"
            return
        
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Use absolute path to the model file
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                self.backend = "fallback"
                return

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False)
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            print("Successfully initialized MediaPipe Pose Landmarker")
            
        except Exception as e:
            print(f"Error initializing Pose Landmarker: {e}")
            print("Pose detection will be skipped")
            self.pose_landmarker = None
            self.backend = "fallback"
    
    def convert_to_openpose_format(self, landmarks, image_width, image_height):
        """
        Convert MediaPipe landmarks (list of NormalizedLandmark) to OpenPose BODY_25 format (25 points)
        using only x, y coordinates.
        """
        openpose_keypoints = np.zeros((25, 3), dtype=np.float32)
        
        mp_to_openpose = {
            0: 0,   # Nose
            11: 2,  # Right shoulder
            13: 3,  # Right elbow
            15: 4,  # Right wrist
            12: 5,  # Left shoulder
            14: 6,  # Left elbow
            16: 7,  # Left wrist
            23: 9,  # Right hip
            25: 10, # Right knee
            27: 11, # Right ankle
            24: 12, # Left hip
            26: 13, # Left knee
            28: 14, # Left ankle
            5: 15,  # Right eye
            2: 16,  # Left eye
            8: 17,  # Right ear
            7: 18,  # Left ear
            32: 19, # Left big toe
            30: 21, # Left heel
            31: 22, # Right big toe
            29: 24, # Right heel
        }
        
        for mp_idx, op_idx in mp_to_openpose.items():
            if mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                openpose_keypoints[op_idx] = [
                    lm.x * image_width,
                    lm.y * image_height,
                    lm.visibility
                ]
        
        # Neck (midpoint of shoulders)
        if len(landmarks) > 12 and landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5:
            neck_x = (landmarks[11].x + landmarks[12].x) / 2 * image_width
            neck_y = (landmarks[11].y + landmarks[12].y) / 2 * image_height
            openpose_keypoints[1] = [neck_x, neck_y, 0.9]
        
        # Mid hip (midpoint of hips)
        if len(landmarks) > 24 and landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5:
            hip_x = (landmarks[23].x + landmarks[24].x) / 2 * image_width
            hip_y = (landmarks[23].y + landmarks[24].y) / 2 * image_height
            openpose_keypoints[8] = [hip_x, hip_y, 0.9]
        
        return openpose_keypoints.flatten().tolist()
    
    def extract_from_video(self, video_path, output_json_dir):
        """Extract pose landmarks from video"""
        if self.pose_landmarker is None:
            print("Error: Pose model not initialized")
            return
        
        os.makedirs(output_json_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        import mediapipe as mp
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use new API exclusively
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.pose_landmarker.detect(mp_image)
            pose_landmarks_list = detection_result.pose_landmarks
            
            if pose_landmarks_list:
                pose_landmarks = pose_landmarks_list[0] # List of NormalizedLandmark
                h, w = frame.shape[:2]
                keypoints = self.convert_to_openpose_format(pose_landmarks, w, h)
                json_data = {"version": 1.3, "people": [{"pose_keypoints_2d": keypoints}]}
            else:
                json_data = {"version": 1.3, "people": []}
            
            json_path = os.path.join(output_json_dir, f"frame_{frame_idx:06d}_keypoints.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            
            frame_idx += 1
            print(f"Processed frame {frame_idx}", end='\r')
        
        cap.release()
        print(f"\nExtracted {frame_idx} frames to {output_json_dir}")



output_json_dir = "outputs/openpose_keypoints"


class SkeletonProcessor:
    EPS = 1e-6

    J = {
        "head": 0,
        "neck": 1,
        "r_shoulder": 2,
        "r_elbow": 3,
        "r_hand": 4,
        "l_shoulder": 5,
        "l_elbow": 6,
        "l_hand": 7,
        "torso": 8,
        "r_hip": 9,
        "r_knee": 10,
        "r_foot": 11,
        "l_hip": 12,
        "l_knee": 13,
        "l_foot": 14,
    }

    def __init__(self, config=None):
        if config is None:
            config = {
                'center_joint': 8,
                'smooth_temporal': False,
                'fix_sequence_length': 97,
                'use_velocity': True,
                'use_acceleration': True,
                'use_angles': True,
                'use_distances': True,
                'use_areas': True,
            }
        self.config = config

    # -----------------------------
    # Helper methods
    # -----------------------------
    @staticmethod
    def _angle(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.sum(ba * bc, axis=-1) / (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + SkeletonProcessor.EPS
        )
        return np.arccos(np.clip(cos, -1.0, 1.0))

    @staticmethod
    def _distance(a, b):
        return np.linalg.norm(a - b, axis=-1)

    @staticmethod
    def _triangle_area(a, b, c):
        return 0.5 * np.abs(
            (b[...,0]-a[...,0])*(c[...,1]-a[...,1]) -
            (c[...,0]-a[...,0])*(b[...,1]-a[...,1])
        )

    # -----------------------------
    # Preprocessing steps
    # -----------------------------
    @staticmethod
    def interpolate_missing(coords, conf, th=0.55):
        T, V, C = coords.shape
        t = np.arange(T)

        for v in range(V):
            valid = conf[:, v] >= th
            if valid.sum() < 2:
                continue
            for c in range(C):
                coords[:, v, c] = np.interp(
                    t, t[valid], coords[valid, v, c]
                )
            conf[~valid, v] *= 0.5
        return coords, conf

    @staticmethod
    def center_skeleton(data, center_joint=8):
        data = data.copy()
        T, V, C = data.shape
        if center_joint >= V:
            raise ValueError(f"center_joint={center_joint} but skeleton only has {V} joints")
        center = data[:, center_joint, :]
        data -= center[:, None, :]
        return data

    @staticmethod
    def normalize_skeleton(data, cfg, left_shoulder=5, right_shoulder=2):
        data = data.copy()
        shoulder_widths = np.linalg.norm(
            data[:, right_shoulder, :] - data[:, left_shoulder, :],
            axis=-1
        )
        scale = np.median(shoulder_widths[shoulder_widths > 0.01])
        return data / scale

    @staticmethod
    def resize_sequence(data, conf, target_T):
        T, V, C = data.shape
        if T == target_T:
            return data, conf

        t_old = np.arange(T)
        t_new = np.linspace(0, T - 1, target_T)

        f_data = interp1d(t_old, data, axis=0, kind='linear', fill_value='extrapolate')
        out = f_data(t_new)

        nearest_indices = np.round(t_new).astype(int)
        nearest_indices = np.clip(nearest_indices, 0, T - 1)
        conf_out = conf[nearest_indices]

        return out, conf_out

    # -----------------------------
    # Feature extraction
    # -----------------------------
    def angle_features(self, coords):
        f = []
        f.append(self._angle(coords[:,5], coords[:,1], coords[:,2]))
        f.append(self._angle(coords[:,5], coords[:,1], coords[:,6]))
        f.append(self._angle(coords[:,2], coords[:,1], coords[:,3]))
        f.append(self._angle(coords[:,12], coords[:,8], coords[:,13]))
        f.append(self._angle(coords[:,9], coords[:,8], coords[:,10]))
        f.append(self._angle(coords[:,5], coords[:,6], coords[:,7]))
        f.append(self._angle(coords[:,2], coords[:,3], coords[:,4]))
        f.append(self._angle(coords[:,12], coords[:,13], coords[:,14]))
        f.append(self._angle(coords[:,9], coords[:,10], coords[:,11]))
        f.append(self._angle(coords[:,0], coords[:,1], coords[:,8]))
        return np.stack(f, axis=1)

    def distance_features(self, coords):
        f = []
        f.append(self._distance(coords[:,7], coords[:,8]))
        f.append(self._distance(coords[:,4], coords[:,8]))
        f.append(self._distance(coords[:,7], coords[:,5]))
        f.append(self._distance(coords[:,4], coords[:,2]))
        f.append(self._distance(coords[:,7], coords[:,12]))
        f.append(self._distance(coords[:,4], coords[:,9]))
        f.append(self._distance(coords[:,7], coords[:,1]))
        f.append(self._distance(coords[:,4], coords[:,1]))
        f.append(self._distance(coords[:,6], coords[:,8]))
        f.append(self._distance(coords[:,3], coords[:,8]))
        f.append(self._distance(coords[:,14], coords[:,11]))
        return np.stack(f, axis=1)

    def area_features(self, coords):
        f = []
        f.append(self._triangle_area(coords[:,7], coords[:,1], coords[:,4]))
        f.append(self._triangle_area(coords[:,5], coords[:,1], coords[:,2]))
        f.append(self._triangle_area(coords[:,7], coords[:,8], coords[:,4]))
        f.append(self._triangle_area(coords[:,6], coords[:,1], coords[:,3]))
        f.append(self._triangle_area(coords[:,14], coords[:,8], coords[:,11]))
        f.append(self._triangle_area(coords[:,13], coords[:,1], coords[:,10]))
        return np.stack(f, axis=1)

    @staticmethod
    def velocity_features(coords):
        vel = np.zeros_like(coords)
        vel[1:-1] = (coords[2:] - coords[:-2]) / 2.0
        vel[0] = coords[1] - coords[0]
        vel[-1] = coords[-1] - coords[-2]
        joints = list(range(18)) 
        mags = [np.linalg.norm(vel[:, j], axis=-1) for j in joints]
        return np.stack(mags, axis=1)

    @staticmethod
    def acceleration_features(coords):
        vel = np.zeros_like(coords)
        vel[1:-1] = (coords[2:] - coords[:-2]) / 2.0
        vel[0] = coords[1] - coords[0]
        vel[-1] = coords[-1] - coords[-2]

        acc = np.zeros_like(coords)
        acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
        acc[0] = vel[1] - vel[0]
        acc[-1] = vel[-1] - vel[-2]

        joints = list(range(18)) 
        mags = [np.linalg.norm(acc[:, j], axis=-1) for j in joints]
        return np.stack(mags, axis=1)

    # -----------------------------
    # Main preprocessing method
    # -----------------------------
    def preprocess(self, skeleton_data):
        T, V, C = skeleton_data.shape

        coords = skeleton_data[:, :, :2].copy()
        confidence = skeleton_data[:, :, 2].copy()

        coords, confidence = self.interpolate_missing(coords, confidence, th=0.55)

        if self.config.get('smooth_temporal', False):
            from scipy.signal import savgol_filter
            for c in range(2):
                coords[:, :, c] = savgol_filter(coords[:, :, c], window_length=3, polyorder=2, axis=0)

        coords = self.center_skeleton(coords, center_joint=self.config['center_joint'])
        coords = self.normalize_skeleton(coords, self.config)

        target_length = self.config.get('fix_sequence_length')
        if target_length is not None and T != target_length:
            coords, confidence = self.resize_sequence(coords, confidence, target_length)
            T = target_length

        # Feature extraction
        feature_blocks = []

        if self.config.get('use_angles', True):
            feature_blocks.append(self.angle_features(coords))
        if self.config.get('use_distances', True):
            feature_blocks.append(self.distance_features(coords))
        if self.config.get('use_areas', True):
            feature_blocks.append(self.area_features(coords))
        if self.config.get('use_velocity', True):
            feature_blocks.append(self.velocity_features(coords))
        if self.config.get('use_acceleration', True):
            feature_blocks.append(self.acceleration_features(coords))

        features = np.concatenate(feature_blocks, axis=1)
        return features
    
    @staticmethod
    def temporal_stats(features, splits=3):
        chunks = np.array_split(features, splits, axis=0)

        out = []
        for c in chunks:
            out.append(c.mean(axis=0))
            out.append(c.std(axis=0))
            out.append(np.median(c, axis=0))

            out.append(c.max(axis=0) - c.min(axis=0))
            out.append(np.mean(c**2, axis=0))
            
        return np.concatenate(out)

processor = SkeletonProcessor()




class EmotionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class BodyEmotionRecognizer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load model
        model_path = os.path.join(base_dir, "emotion_model.pth")
        checkpoint = torch.load(model_path, map_location = 'cpu')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionMLP(
            input_dim = checkpoint['input_dim'],
            num_classes = checkpoint['num_classes']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        scaler_path = os.path.join(base_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load label encoder
        le_path = os.path.join(base_dir, "label_encoder.pkl")
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)
        
        # Emotion labels
        self.label_map = checkpoint['emotion_names']
        print('Loaded label_map:', self.label_map, flush=True)
        
        # Configuration
        self.target_frames = 97
        self.chunks_root = "outputs/chunks"
        self.output_json_dir = "outputs/openpose_keypoints"  

    @staticmethod
    def create_chunks(json_files, window_size = 97, output_root = "outputs/chunks"):
        os.makedirs(output_root, exist_ok=True)

        chunks = []
        chunk_id = 1

        for i in range(0, len(json_files), window_size):

            chunk = json_files[i:i + window_size]

            # Repeat last frame)
            if len(chunk) < window_size and len(chunk) > 0:
                pad_count = window_size - len(chunk)
                chunk.extend([chunk[-1]] * pad_count)

            if len(chunk) == 0:
                continue

            folder = os.path.join(output_root, f"chunk_{chunk_id:03d}")
            os.makedirs(folder, exist_ok=True)

            for f in chunk:
                shutil.copy(f, folder)

            chunks.append(folder)
            print(f"Saved chunk {chunk_id} with {len(chunk)} frames")

            chunk_id += 1
    
    def load_chunk_skeleton(self, chunk_path, target_frames):
        json_files = sorted(glob.glob(os.path.join(chunk_path, "*.json")))

        if len(json_files) == 0:
            return np.zeros((target_frames, 25, 3), dtype = np.float32)

        video_skeleton = []

        for jf in json_files[:target_frames]:

            try:
                with open(jf) as f:
                    data = json.load(f)

                if len(data["people"]) == 0:
                    kp = np.zeros((25, 3), dtype = np.float32)

                else:
                    kp_raw = data["people"][0]["pose_keypoints_2d"]

                    # BODY_25 format → 75 values
                    kp = np.array(kp_raw).reshape(25, 3)

                video_skeleton.append(kp)

            except Exception:
                video_skeleton.append(np.zeros((25, 3), dtype = np.float32))


        return np.stack(video_skeleton, axis = 0)
    
    def load_all_chunks(self, chunks_root = None, target_frames = None):
        if chunks_root is None:
            chunks_root = self.chunks_root
        if target_frames is None:
            target_frames = self.target_frames
        
        # Get all chunk folders
        chunk_paths = sorted(glob.glob(os.path.join(chunks_root, "chunk_*")))
        print(f"Found chunks: {len(chunk_paths)}")
        
        all_skeletons = []
        
        for chunk_path in chunk_paths:
            skel = self.load_chunk_skeleton(chunk_path, target_frames)
            skel = processor.preprocess(skel)  # (T, F)
            skel = processor.temporal_stats(skel)    # (F * 5 * 3)
            all_skeletons.append(skel)
        
        if not all_skeletons:
            return np.array([])
        
        return np.stack(all_skeletons, axis=0)
    
    def predict_chunks(self, X_test):
        if len(X_test) == 0:
            return []
        
        # Normalize features
        X_scaled = self.scaler.transform(X_test)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        emotion_predictions = self.le.inverse_transform(predictions)
        
        return emotion_predictions.tolist()
    
    def predict(self, image_windows):
        if self.model is None:
            print("Model not loaded, returning neutral predictions")
            return ["Neutral"] * len(image_windows)
            
        json_files = sorted(glob.glob(f"{self.output_json_dir}/*.json"))
        print(f"Total frames extracted: {len(json_files)}")
        
        if not json_files:
            print("No JSON files found, returning neutral predictions")
            return ["Neutral"] * len(image_windows)
       
        BodyEmotionRecognizer.create_chunks(json_files, window_size = self.target_frames)
        
        X_test = self.load_all_chunks()
        
        if len(X_test) == 0:
            print("No valid chunks loaded, returning neutral predictions")
            return ["Neutral"] * len(image_windows)
        
        # Get predictions
        predictions = self.predict_chunks(X_test)
    
        # Forward fill predictions if fewer chunks than windows
        if predictions:
            if len(predictions) < len(image_windows):
                last_pred = predictions[-1]
                predictions.extend([last_pred] * (len(image_windows) - len(predictions)))
            elif len(predictions) > len(image_windows):
                predictions = predictions[:len(image_windows)]
        else:
            predictions = ["Neutral"] * len(image_windows)
        
        return predictions
    
    def predict_single_chunk(self, chunk_path):
        skel = self.load_chunk_skeleton(chunk_path, self.target_frames)
        skel = processor.preprocess(skel)
        skel = processor.temporal_stats(skel)
        
        # Reshape for single prediction
        X = skel.reshape(1, -1)
        
        predictions = self.predict_chunks(X)
        return predictions[0] if predictions else "gg5"

