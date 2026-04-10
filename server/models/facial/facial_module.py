import torch
import os
import torch.nn as nn
import cv2
from torchvision import transforms
from collections import Counter


class GiMeFive(nn.Module):
    def __init__(self, num_classes=6, dropout_conv=0.2, dropout_fc=0.5):
        super(GiMeFive, self).__init__()

        def conv_block(in_ch, out_ch, dropout):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout)
            )

        self.features = nn.Sequential(
            conv_block(3,    64,   dropout_conv),
            conv_block(64,   128,  dropout_conv),
            conv_block(128,  256,  dropout_conv),
            conv_block(256,  512,  dropout_conv),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FacialEmotionRecognizer:

    # Calibrated for GiMeFive trained on RAF-DB (6 classes, no neutral):
    #   - Floor set below weakest class floor (fear ~0.38 when correct)
    #   - Entropy catches flat/spread distributions neutral faces produce
    CONFIDENCE_THRESHOLD = 0.57
    ENTROPY_THRESHOLD    = 0.88
    _LOG6 = torch.log(torch.tensor(6.0))

    def __init__(self):
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best_model_run16_drw_soft.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GiMeFive(num_classes=6, dropout_conv=0.2, dropout_fc=0.5).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.label_map = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    def _is_neutral(self, logits: torch.Tensor) -> bool:
        probs = torch.softmax(logits, dim=1)[0]

        if probs.max().item() < self.CONFIDENCE_THRESHOLD:
            return True

        entropy      = -(probs * torch.log(probs + 1e-9)).sum()
        norm_entropy = (entropy / self._LOG6.to(self.device)).item()
        if norm_entropy > self.ENTROPY_THRESHOLD:
            return True

        return False

    def predict(self, image_windows):
        window_predictions = []

        for group in image_windows:
            emotions = []

            for frame_path in group:
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_cascade = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_detector.detectMultiScale(gray_cascade, 1.3, 5)
                if len(faces) == 0:
                    continue

                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face = frame_rgb[y:y+h, x:x+w]

                img = self.transform(face).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(img)

                    if self._is_neutral(out):
                        emotions.append("neutral")
                        continue

                    idx = torch.argmax(out, dim=1).item()
                    emotions.append(self.label_map[idx])

            window_predictions.append(
                Counter(emotions).most_common(1)[0][0] if emotions else "no_face"
            )

        return window_predictions