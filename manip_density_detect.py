# manip_density_detect.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2


# ---------------------------
# UNetDensity (eğitimdekiyle aynı)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetDensity(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = ConvBlock(1, 32)
        self.d2 = ConvBlock(32, 64)
        self.b  = ConvBlock(64, 128)
        self.u2 = ConvBlock(128+64, 64)
        self.u1 = ConvBlock(64+32, 32)
        self.den_head = nn.Conv2d(32, 1, 1)
        self.msk_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, 2))
        b  = self.b(F.max_pool2d(d2, 2))
        u2 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, d2], 1))
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, d1], 1))
        pred_den = torch.sigmoid(self.den_head(u1))   # [0,1]
        pred_msk_logit = self.msk_head(u1)
        return pred_den, pred_msk_logit

# ---------------------------
# Model cache (API için şart)
# ---------------------------
_MODEL = None
_MODEL_PATH = None
_DEVICE_STR = None

def load_model_once(model_path: str, device=None):
    global _MODEL, _MODEL_PATH, _DEVICE_STR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_str = str(device)

    if _MODEL is not None and _MODEL_PATH == model_path and _DEVICE_STR == dev_str:
        return _MODEL, device

    m = UNetDensity().to(device)
    sd = torch.load(model_path, map_location=device)
    m.load_state_dict(sd)
    m.eval()

    _MODEL = m
    _MODEL_PATH = model_path
    _DEVICE_STR = dev_str
    return _MODEL, device

# ---------------------------
# Utils
# ---------------------------
def preprocess_gray_128(image_path: str, img_size=128):
    img = Image.open(image_path).convert("L").resize((img_size, img_size))
    arr01 = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr01).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return x, arr01

def density_to_boxes(d01: np.ndarray, thr=0.28, min_area=20):
    m = (d01 >= thr).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append((int(x), int(y), int(x+w), int(y+h)))
    return boxes

@torch.no_grad()
def detect_manipulation_density(
    model_path: str,
    image_path: str,
    thr: float = 0.28,
    min_area: int = 20,
    return_density_u8: bool = True
):
    """
    Returns:
      - boxes_xyxy on 128x128
      - density01 float map [H,W]
      - density_u8 (optional) 0-255
    """
    model, device = load_model_once(model_path)

    x, _ = preprocess_gray_128(image_path, img_size=128)
    x = x.to(device)

    pred_den, _, _= model(x)
    den = pred_den[0,0].detach().cpu().numpy().astype(np.float32)  # [H,W] in [0,1]
    den01 = den / (den.max() + 1e-8)

    boxes = density_to_boxes(den01, thr=thr, min_area=min_area)

    out = {
        "image_path": os.path.abspath(image_path),
        "device": str(device),
        "thr": float(thr),
        "min_area": int(min_area),
        "boxes_xyxy": boxes,
        "count": int(len(boxes)),
        "is_manipulated": bool(len(boxes) > 0),
        "density_max": float(den.max()),
        "density_mean": float(den.mean()),
        "density01": den01,  # float32
    }

    if return_density_u8:
        out["density_u8"] = (den01 * 255).clip(0,255).astype(np.uint8)

    return out
