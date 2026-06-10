import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import random

IMG_SIZE = 128

def read_gray_u8(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)

def read_den_f32(path):
    den = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if den is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    if den.shape[:2] != (IMG_SIZE, IMG_SIZE):
        den = cv2.resize(den, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    den = den.astype(np.float32) / 255.0
    return den

def read_mask01(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    if m.shape[:2] != (IMG_SIZE, IMG_SIZE):
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)

def read_boxes_from_yolo(lbl_path):
    boxes=[]
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            _, xc, yc, bw, bh = map(float, p)
            x1 = int((xc - bw/2) * IMG_SIZE)
            y1 = int((yc - bh/2) * IMG_SIZE)
            x2 = int((xc + bw/2) * IMG_SIZE)
            y2 = int((yc + bh/2) * IMG_SIZE)
            x1 = max(0, min(IMG_SIZE-1, x1)); y1 = max(0, min(IMG_SIZE-1, y1))
            x2 = max(0, min(IMG_SIZE-1, x2)); y2 = max(0, min(IMG_SIZE-1, y2))
            if x2 > x1 and y2 > y1:
                boxes.append((x1,y1,x2,y2))
    return boxes

class ManipDensityDataset(Dataset):
    def __init__(self, ids, img_dir, den_dir, msk_dir, lbl_dir, augment=False):
        self.ids = ids
        self.augment = augment
        self.img_dir = img_dir
        self.den_dir = den_dir
        self.msk_dir = msk_dir
        self.lbl_dir = lbl_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img = read_gray_u8(os.path.join(self.img_dir, f"{sid}.png"))
        den = read_den_f32(os.path.join(self.den_dir, f"{sid}.png")) if os.path.isdir(self.den_dir) else np.zeros((IMG_SIZE,IMG_SIZE),np.float32)
        msk = read_mask01(os.path.join(self.msk_dir, f"{sid}.png")) if os.path.isdir(self.msk_dir) else (den>0).astype(np.float32)
        boxes = read_boxes_from_yolo(os.path.join(self.lbl_dir, f"{sid}.txt")) if os.path.isdir(self.lbl_dir) else []

        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0     # [1,H,W]
        den_t = torch.from_numpy(den).float().unsqueeze(0)             # [1,H,W] in [0,1]
        msk_t = torch.from_numpy(msk).float().unsqueeze(0)             # [1,H,W] in {0,1}

        if self.augment:
            if random.random() < 0.5:
                img_t = TF.hflip(img_t); den_t = TF.hflip(den_t); msk_t = TF.hflip(msk_t)
                boxes = [(IMG_SIZE-x2, y1, IMG_SIZE-x1, y2) for (x1,y1,x2,y2) in boxes]
            if random.random() < 0.3:
                img_t = TF.vflip(img_t); den_t = TF.vflip(den_t); msk_t = TF.vflip(msk_t)
                boxes = [(x1, IMG_SIZE-y2, x2, IMG_SIZE-y1) for (x1,y1,x2,y2) in boxes]

        return {"id": sid, "img": img_t, "den": den_t, "msk": msk_t, "boxes": boxes}

def collate_keep_boxes(batch):
    return {
        "id":   [b["id"] for b in batch],
        "img":  torch.stack([b["img"] for b in batch], 0),
        "den":  torch.stack([b["den"] for b in batch], 0),
        "msk":  torch.stack([b["msk"] for b in batch], 0),
        "boxes":[b["boxes"] for b in batch],
    }
