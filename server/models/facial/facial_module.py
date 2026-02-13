import torch
import os
import torch.nn as nn
import cv2
import math
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision import transforms
from collections import Counter

# ============ MODEL BLOCKS (UNCHANGED FROM COLAB) ============
class DepthwiseConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        padding = 0 if stride > 1 else kernel_size // 2
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.bn1(self.depth(x)))
        x = self.act(self.bn2(self.point(x)))
        return x

class DetailExtractionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.sep4 = SeparableConv2d(channels, channels, 4, 4)
        self.sep2 = SeparableConv2d(channels, channels, 2, 2)
        self.drop = nn.Dropout2d(0.1)
        self.ln2 = nn.LayerNorm(channels)
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.ln1(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.sep4(x)
        x = self.sep2(x)
        x = self.drop(x)
        pooled = self.ln2(x.mean(dim=(2,3)))
        q = pooled.unsqueeze(2)
        k = pooled.unsqueeze(1)
        attn = torch.softmax(torch.matmul(q,k)/math.sqrt(pooled.shape[-1]), dim=-1)
        v = pooled.unsqueeze(2)
        return torch.matmul(attn,v).squeeze(2)

class ConvCut(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.features.children())[:6])
        for p in self.backbone.parameters(): p.requires_grad=False
        for p in self.backbone[-2:].parameters(): p.requires_grad=True
        self.C=512
        self.pre=DepthwiseConv(self.C)
        self.ln1=nn.LayerNorm(self.C)
        self.det=DetailExtractionBlock(self.C)
        self.ln2=nn.LayerNorm(self.C)
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(self.C, num_classes)
    def forward(self,x):
        x=self.backbone(x)
        x=self.pre(x)
        x=self.ln1(x.permute(0,2,3,1)).permute(0,3,1,2)
        x=self.det(x)
        x=self.ln2(x)
        x=self.dropout(x)
        return self.fc(x)

# ============ MAIN CLASS ============
class FacialEmotionRecognizer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "convcut_vanilla+hardF.pth")
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=ConvCut(8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self.model.load_state_dict(torch.load("models/facial/convcut_fer.pth", map_location=self.device))
        self.model.eval()

        self.face_detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

        self.transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        self.label_map=['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']

    def predict(self, image_windows):
        window_predictions=[]
        for group in image_windows:
            emotions=[]
            for frame_path in group:
                frame=cv2.imread(frame_path)
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=self.face_detector.detectMultiScale(gray,1.3,5)
                if len(faces)==0: continue
                x,y,w,h=max(faces,key=lambda f:f[2]*f[3])
                face=frame[y:y+h,x:x+w]
                img=self.transform(face).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out=self.model(img)
                    idx=torch.argmax(out,dim=1).item()
                emotions.append(self.label_map[idx])
            window_predictions.append(Counter(emotions).most_common(1)[0][0] if emotions else "no_face")
        return window_predictions
