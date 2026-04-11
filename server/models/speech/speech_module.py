# import numpy as np
# import os
# import librosa
# import joblib
# import pickle
# from tensorflow.keras.models import load_model


# TARGET_SR = 22050
# MIN_AUDIO_LENGTH = 2048


# class SpeechEmotionRecognizer:

#     def __init__(self):
#         # Get the absolute path to this file's directory
#         speech_dir = os.path.dirname(os.path.abspath(__file__))

#         # Build absolute paths to model files
#         model_path = os.path.join(speech_dir, "speech_emotion_model.keras")
#         scaler_path = os.path.join(speech_dir, "scaler.save")
#         labels_path = os.path.join(speech_dir, "label_mapping.pkl")

#         # Check if files exist and provide helpful error messages
#         missing_files = []
#         for path in [model_path, scaler_path, labels_path]:
#             if not os.path.exists(path):
#                 missing_files.append(os.path.basename(path))
        
#         if missing_files:
#             raise FileNotFoundError(
#                 f"Missing required files in {speech_dir}:\n" +
#                 "\n".join(f"  - {f}" for f in missing_files) +
#                 f"\n\nExpected location: {speech_dir}"
#             )

#         # Load model and preprocessing files
#         try:
#             self.model = load_model(model_path)
#             self.scaler = joblib.load(scaler_path)
#             with open(labels_path, "rb") as f:
#                 self.lb = pickle.load(f)
#             print(f"Speech emotion model loaded successfully from {speech_dir}")
#         except Exception as e:
#             raise RuntimeError(f"Error loading model files: {e}")

#     # =========================================================
#     # SILENCE REMOVAL (Paper Logic)
#     # =========================================================
#     def custom_silence_removal(self, y, sr):
#         intervals = librosa.effects.split(y, top_db=30)
#         if len(intervals) == 0:
#             return y

#         start_speech_idx = intervals[0][0]
#         end_speech_idx = intervals[-1][1]

#         start_silence_ms = (start_speech_idx / sr) * 1000
#         end_silence_ms = ((len(y) - end_speech_idx) / sr) * 1000

#         new_start, new_end = 0, len(y)

#         if start_silence_ms > 200:
#             new_start = int(start_speech_idx * 0.70)

#         if end_silence_ms > 200:
#             silence_samples = len(y) - end_speech_idx
#             new_end = len(y) - int(silence_samples * 0.70)

#         return y[new_start:new_end]

#     # =========================================================
#     # FEATURE EXTRACTION (190 FEATURES)
#     # =========================================================
#     def extract_features_paper(self, y):
#         max_n_fft = min(2048, len(y))
#         n_fft_melspec = max_n_fft
#         n_fft_chroma = min(1024, len(y))

#         features = []

#         mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=20, n_fft=n_fft_melspec)
#         mfccs_delta = librosa.feature.delta(mfccs)
#         mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

#         features.extend(np.mean(mfccs, axis=1))
#         features.extend(np.mean(mfccs_delta, axis=1))
#         features.extend(np.mean(mfccs_delta2, axis=1))
#         features.extend(np.std(mfccs, axis=1))

#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=TARGET_SR, n_chroma=12, n_fft=n_fft_chroma)
#         chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=TARGET_SR, n_chroma=12)
#         chroma_cens = librosa.feature.chroma_cens(y=y, sr=TARGET_SR, n_chroma=12)

#         features.extend(np.mean(chroma_stft, axis=1))
#         features.extend(np.mean(chroma_cqt, axis=1))
#         features.extend(np.mean(chroma_cens, axis=1))

#         melspec = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=64, n_fft=n_fft_melspec)
#         log_mel = librosa.power_to_db(melspec)
#         features.extend(np.mean(log_mel, axis=1))

#         spec_contrast = librosa.feature.spectral_contrast(y=y, sr=TARGET_SR, n_bands=5)
#         features.extend(np.mean(spec_contrast, axis=1))

#         rmse = librosa.feature.rms(y=y)
#         features.extend([np.mean(rmse), np.std(rmse), np.max(rmse)])

#         zcr = librosa.feature.zero_crossing_rate(y)
#         features.append(np.mean(zcr))

#         return np.array(features)

#     # =========================================================
#     # PREDICT PER WINDOW - ACCEPTS PRE-WINDOWED AUDIO
#     # =========================================================
#     def predict(self, audio_windows, sr):
#         """
#         Predict emotions for pre-windowed audio segments.
        
#         Args:
#             audio_windows: List of numpy arrays (audio windows from ai_pipeline)
#             sr: Sample rate of the audio
        
#         Returns:
#             List of predicted emotion labels
#         """
#         if not audio_windows:
#             return []

#         features_list = []

#         for window in audio_windows:
#             # Apply silence removal to each window
#             window = self.custom_silence_removal(window, sr)
            
#             # Resample to TARGET_SR if needed
#             if sr != TARGET_SR:
#                 window = librosa.resample(window, orig_sr=sr, target_sr=TARGET_SR)
            
#             # Ensure minimum length
#             if len(window) < MIN_AUDIO_LENGTH:
#                 window = np.pad(window, (0, MIN_AUDIO_LENGTH - len(window)), mode='constant')

#             # Extract features
#             feat = self.extract_features_paper(window)
#             if len(feat) == 190:
#                 features_list.append(feat)

#         if not features_list:
#             return []

#         # Prepare input for model
#         X_input = np.array(features_list)
#         X_input = self.scaler.transform(X_input)
#         X_input = X_input.reshape(X_input.shape[0], 1, X_input.shape[1])

#         # Predict
#         y_pred_probs = self.model.predict(X_input)
#         y_pred_idx = np.argmax(y_pred_probs, axis=1)
#         y_pred_labels = self.lb.inverse_transform(y_pred_idx)

#         return y_pred_labels.tolist()




import numpy as np
import os
import librosa
import joblib
import pickle
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model


TARGET_SR = 22050
MIN_AUDIO_LENGTH = 4096


class SpeechEmotionRecognizer:

    def __init__(self):
        speech_dir = os.path.dirname(os.path.abspath(__file__))

        model_path  = os.path.join(speech_dir, "speech_emotion_model.keras")
        scaler_path = os.path.join(speech_dir, "scaler.save")
        labels_path = os.path.join(speech_dir, "label_mapping.pkl")
        pca_path    = os.path.join(speech_dir, "pca.pkl")          # ← new

        missing_files = []
        for path in [model_path, scaler_path, labels_path, pca_path]:
            if not os.path.exists(path):
                missing_files.append(os.path.basename(path))

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {speech_dir}:\n" +
                "\n".join(f"  - {f}" for f in missing_files) +
                f"\n\nExpected location: {speech_dir}"
            )

        try:
            self.model  = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(labels_path, "rb") as f:
                self.lb = pickle.load(f)
            with open(pca_path, "rb") as f:          # ← new
                self.pca = pickle.load(f)             # ← new
            print(f"Speech emotion model loaded successfully from {speech_dir}")
        except Exception as e:
            raise RuntimeError(f"Error loading model files: {e}")

    # =========================================================
    # SILENCE REMOVAL (unchanged)
    # =========================================================
    def custom_silence_removal(self, y, sr):
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) == 0:
            return y

        start_speech_idx = intervals[0][0]
        end_speech_idx   = intervals[-1][1]

        start_silence_ms = (start_speech_idx / sr) * 1000
        end_silence_ms   = ((len(y) - end_speech_idx) / sr) * 1000

        new_start, new_end = 0, len(y)
        if start_silence_ms > 200:
            new_start = int(start_speech_idx * 0.70)
        if end_silence_ms > 200:
            silence_samples = len(y) - end_speech_idx
            new_end = len(y) - int(silence_samples * 0.70)

        return y[new_start:new_end]

    # =========================================================
    # FEATURE EXTRACTION — 190 features (unchanged)
    # =========================================================
    def extract_features_paper(self, y):
        max_n_fft      = min(2048, len(y))
        n_fft_melspec  = max_n_fft
        n_fft_chroma   = min(1024, len(y))

        features = []

        mfccs         = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=20, n_fft=n_fft_melspec)
        
        if mfccs.shape[1] < 9:
            pad_amount = 9 - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_amount)), mode='constant')
        
        mfccs_delta   = librosa.feature.delta(mfccs)
        mfccs_delta2  = librosa.feature.delta(mfccs, order=2)
        features.extend(np.mean(mfccs,        axis=1))
        features.extend(np.mean(mfccs_delta,  axis=1))
        features.extend(np.mean(mfccs_delta2, axis=1))
        features.extend(np.std(mfccs,         axis=1))

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=TARGET_SR, n_chroma=12, n_fft=n_fft_chroma)
        chroma_cqt  = librosa.feature.chroma_cqt(y=y, sr=TARGET_SR, n_chroma=12)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=TARGET_SR, n_chroma=12)
        features.extend(np.mean(chroma_stft, axis=1))
        features.extend(np.mean(chroma_cqt,  axis=1))
        features.extend(np.mean(chroma_cens, axis=1))

        melspec  = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=64, n_fft=n_fft_melspec)
        log_mel  = librosa.power_to_db(melspec)
        features.extend(np.mean(log_mel, axis=1))

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=TARGET_SR, n_bands=5)
        features.extend(np.mean(spec_contrast, axis=1))

        rmse = librosa.feature.rms(y=y)
        features.extend([np.mean(rmse), np.std(rmse), np.max(rmse)])

        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))

        return np.array(features)

    # =========================================================
    # PREDICT — now applies PCA after feature extraction
    # =========================================================
    def predict(self, audio_windows, sr):
        if not audio_windows:
            return []

        features_list = []
        for window in audio_windows:
            window = self.custom_silence_removal(window, sr)
            if sr != TARGET_SR:
                window = librosa.resample(window, orig_sr=sr, target_sr=TARGET_SR)
            if len(window) < MIN_AUDIO_LENGTH:
                window = np.pad(window, (0, MIN_AUDIO_LENGTH - len(window)), mode='constant')

            feat = self.extract_features_paper(window)
            if len(feat) == 190:
                features_list.append(feat)

        if not features_list:
            return []

        X_input = np.array(features_list)          # (N, 190)
        X_input = self.scaler.transform(X_input)   # scale first  ← matches training step 1
        X_input = self.pca.transform(X_input)      # then PCA     ← matches training step 2
        X_input = X_input.reshape(X_input.shape[0], 1, X_input.shape[1])

        y_pred_probs  = self.model.predict(X_input)
        y_pred_idx    = np.argmax(y_pred_probs, axis=1)
        y_pred_labels = self.lb.inverse_transform(y_pred_idx)

        return y_pred_labels.tolist()
