"""
현미경 원형 시료 데이터 생성 및 전처리 모듈

핵심 개선점:
1. 실제 데이터 도메인 반영 (r_norm: 0.38~0.50, 어두운 배경, 밝은 원)
2. 선명한 테두리 강조
3. 강력한 augmentation
"""

import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 256
_COORD_CACHE = {}


def _edge_channel(img_rgb):
    """Sobel 기반 edge 검출 (원형 테두리 강조)"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mag = mag / (mag.max() + 1e-6)
    return mag.astype(np.float32)


def _coord_channels(height, width):
    """좌표 채널 생성 (위치 정보 제공)"""
    key = (height, width)
    cached = _COORD_CACHE.get(key)
    if cached is not None:
        return cached
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv, yv], axis=0)
    _COORD_CACHE[key] = coords
    return coords


def preprocess_image(img_rgb):
    """RGB uint8 -> (6, H, W) float32 (RGB + edge + coord)"""
    img = img_rgb.astype(np.float32) / 255.0
    edge = _edge_channel(img_rgb)[..., None]
    coords = _coord_channels(img_rgb.shape[0], img_rgb.shape[1]).transpose(1, 2, 0)
    merged = np.concatenate([img, edge, coords], axis=2)
    return merged.transpose(2, 0, 1).astype(np.float32)


def _apply_intensity_aug(img, strength=1.0):
    """밝기/대비 augmentation"""
    if np.random.rand() < 0.7 * strength:
        brightness = np.random.uniform(-0.12, 0.12)
        contrast = np.random.uniform(0.8, 1.25)
        img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0, 1)
    return img


def _apply_noise_blur(img, strength=1.0):
    """노이즈 및 블러 augmentation"""
    if np.random.rand() < 0.3 * strength:
        blur_size = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur((img * 255).astype(np.uint8), (blur_size, blur_size), 0).astype(np.float32) / 255.0
    if np.random.rand() < 0.5 * strength:
        noise = np.random.normal(0, np.random.uniform(0.008, 0.025), img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 1)
    return img


def _apply_shadow(img):
    """그림자 효과"""
    if np.random.rand() < 0.4:
        h, w = img.shape[:2]
        x0 = np.random.uniform(-0.2, 0.6) * w
        y0 = np.random.uniform(-0.2, 0.6) * h
        x1 = x0 + np.random.uniform(0.4, 1.0) * w
        y1 = y0 + np.random.uniform(0.4, 1.0) * h
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 1, -1)
        alpha = np.random.uniform(0.05, 0.18)
        img = np.clip(img * (1 - alpha * mask[..., None]), 0, 1)
    return img


def _apply_vignette(img, strength=1.0):
    """비네팅 효과 (가장자리 어둡게)"""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y = h / 2
    center_x = w / 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    dist = dist / dist.max()
    vignette_strength = np.random.uniform(0.1, 0.25) * strength
    vignette = 1.0 - vignette_strength * dist
    return np.clip(img * vignette[..., None], 0, 1)


def resize_with_padding(img_rgb, target_size=IMG_SIZE):
    """이미지를 정사각형으로 패딩하며 리사이즈"""
    h, w = img_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h))
    pad_x_left = (target_size - new_w) // 2
    pad_y_top = (target_size - new_h) // 2
    canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
    canvas[pad_y_top:pad_y_top + new_h, pad_x_left:pad_x_left + new_w] = resized
    meta = {
        "scale": scale,
        "pad_x": pad_x_left,
        "pad_y": pad_y_top,
        "orig_w": w,
        "orig_h": h,
        "new_w": new_w,
        "new_h": new_h,
    }
    return canvas, meta


def normalize_label_to_square(x_norm, y_norm, r_norm, meta, target_size=IMG_SIZE):
    """원본 좌표를 정사각형 좌표로 변환"""
    x_px = x_norm * meta["orig_w"]
    y_px = y_norm * meta["orig_h"]
    r_px = r_norm * min(meta["orig_w"], meta["orig_h"])
    x_px = x_px * meta["scale"] + meta["pad_x"]
    y_px = y_px * meta["scale"] + meta["pad_y"]
    r_px = r_px * meta["scale"]
    return (
        np.clip(x_px / target_size, 0, 1),
        np.clip(y_px / target_size, 0, 1),
        np.clip(r_px / target_size, 0, 1),
    )


def denormalize_from_square(pred, meta, target_size=IMG_SIZE):
    """정사각형 좌표를 원본 좌표로 역변환"""
    x_px = pred[0] * target_size
    y_px = pred[1] * target_size
    r_px = pred[2] * target_size
    x_px = (x_px - meta["pad_x"]) / meta["scale"]
    y_px = (y_px - meta["pad_y"]) / meta["scale"]
    r_px = r_px / meta["scale"]
    return int(round(x_px)), int(round(y_px)), int(round(r_px))


class SyntheticCircleDataset(Dataset):
    """
    현미경 원형 시료 합성 데이터셋
    
    실제 데이터 도메인 특성 반영:
    - 어두운 배경 (0.02~0.12)
    - 밝은 원형 시료 (0.7~0.98)
    - 선명한 테두리
    - r_norm 범위: 0.38~0.50 (실제 데이터 분포)
    """
    def __init__(self, num_samples=2000, domain_match=True):
        self.num_samples = num_samples
        self.domain_match = domain_match

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 어두운 배경 (현미경 이미지 특성)
        base = np.random.uniform(0.02, 0.12)
        bg_color = np.array([
            base * np.random.uniform(0.85, 1.15),
            base * np.random.uniform(0.85, 1.15),
            base * np.random.uniform(0.85, 1.2),
        ], dtype=np.float32)
        bg_color = np.clip(bg_color, 0, 0.18)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), bg_color, dtype=np.float32)
        
        # 비네팅 효과
        img = _apply_vignette(img, strength=1.2)
        
        # 원 위치 (중심 근처, 실제 데이터처럼)
        if self.domain_match:
            x_c = np.random.uniform(0.42, 0.58) * IMG_SIZE
            y_c = np.random.uniform(0.42, 0.62) * IMG_SIZE
            r = np.random.uniform(0.38, 0.50) * IMG_SIZE
        else:
            x_c = np.random.uniform(0.15, 0.85) * IMG_SIZE
            y_c = np.random.uniform(0.15, 0.85) * IMG_SIZE
            r = np.random.uniform(0.25, 0.55) * IMG_SIZE
        
        # 밝은 원형 시료 본체
        body_brightness = np.random.uniform(0.75, 0.98)
        body_color = np.array([
            body_brightness * np.random.uniform(0.95, 1.05),
            body_brightness * np.random.uniform(0.92, 1.02),
            body_brightness * np.random.uniform(0.88, 1.0),
        ], dtype=np.float32)
        body_color = np.clip(body_color, 0, 1)
        cv2.circle(img, (int(x_c), int(y_c)), int(r), body_color.tolist(), -1)
        
        # 선명한 테두리 (핵심!)
        if np.random.rand() < 0.9:
            edge_darkness = np.random.uniform(0.5, 0.8)
            edge_color = np.clip(body_color * edge_darkness, 0, 1)
            edge_thickness = int(np.random.choice([3, 4, 5, 6, 7]))
            cv2.circle(img, (int(x_c), int(y_c)), int(r), edge_color.tolist(), edge_thickness)
        
        # 내부 밝은 영역 (시료 특성)
        if np.random.rand() < 0.6:
            inner_r = int(r * np.random.uniform(0.5, 0.8))
            inner_color = np.clip(body_color * np.random.uniform(0.9, 1.05), 0, 1)
            cv2.circle(img, (int(x_c), int(y_c)), inner_r, inner_color.tolist(), -1)
        
        # 시료 내부 점/입자
        num_dots = np.random.randint(20, 60)
        for _ in range(num_dots):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, r * 0.9)
            dot_x = int(x_c + dist * np.cos(angle))
            dot_y = int(y_c + dist * np.sin(angle))
            if 0 <= dot_x < IMG_SIZE and 0 <= dot_y < IMG_SIZE:
                dot_size = int(np.random.randint(1, 5))
                dot_color = (
                    np.random.uniform(0.2, 0.5),
                    np.random.uniform(0.15, 0.45),
                    np.random.uniform(0.2, 0.5),
                )
                cv2.circle(img, (dot_x, dot_y), dot_size, dot_color, -1)
        
        # 배경 노이즈/선
        if np.random.rand() < 0.25:
            for _ in range(np.random.randint(1, 4)):
                x0 = np.random.randint(0, IMG_SIZE)
                y0 = np.random.randint(0, IMG_SIZE)
                x1 = np.random.randint(0, IMG_SIZE)
                y1 = np.random.randint(0, IMG_SIZE)
                color = np.random.uniform(0.08, 0.25)
                cv2.line(img, (x0, y0), (x1, y1), (color, color, color), 1)
        
        # Augmentation
        img = _apply_intensity_aug(img)
        img = _apply_shadow(img)
        img = _apply_noise_blur(img)
        img = np.clip(img, 0, 1)
        
        label = np.array([x_c / IMG_SIZE, y_c / IMG_SIZE, r / IMG_SIZE], dtype=np.float32)
        return torch.tensor(preprocess_image((img * 255).astype(np.uint8))), torch.tensor(label)


class RealImageDataset(Dataset):
    """실제 이미지 데이터셋 (labels.json 기반)"""
    def __init__(self, labels_file, augment=True):
        with open(labels_file, "r") as f:
            labels = json.load(f)
        self.items = []
        for path, info in labels.items():
            if "x_norm" in info and "y_norm" in info and "r_norm" in info:
                label = [float(info["x_norm"]), float(info["y_norm"]), float(info["r_norm"])]
            else:
                width = float(info.get("width", info.get("w", 0)))
                height = float(info.get("height", info.get("h", 0)))
                x_norm = float(info["x"]) / width
                y_norm = float(info["y"]) / height
                r_norm = float(info["r"]) / min(width, height)
                label = [x_norm, y_norm, r_norm]
            self.items.append((path, label))
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, meta = resize_with_padding(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        
        x_norm, y_norm, r_norm = label
        x_norm, y_norm, r_norm = normalize_label_to_square(x_norm, y_norm, r_norm, meta, IMG_SIZE)
        
        if self.augment:
            # 기하학적 augmentation
            if np.random.rand() < 0.7:
                angle = np.random.uniform(-8, 8)
                scale = np.random.uniform(0.92, 1.08)
                shift_x = np.random.uniform(-0.06, 0.06) * IMG_SIZE
                shift_y = np.random.uniform(-0.06, 0.06) * IMG_SIZE
                M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, scale)
                M[0, 2] += shift_x
                M[1, 2] += shift_y
                img = cv2.warpAffine((img * 255).astype(np.uint8), M, (IMG_SIZE, IMG_SIZE), borderValue=(0, 0, 0))
                img = img.astype(np.float32) / 255.0
                point = np.array([x_norm * IMG_SIZE, y_norm * IMG_SIZE, 1.0], dtype=np.float32)
                new_xy = M @ point
                x_norm = np.clip(new_xy[0] / IMG_SIZE, 0, 1)
                y_norm = np.clip(new_xy[1] / IMG_SIZE, 0, 1)
                r_norm = np.clip(r_norm * scale, 0, 1)
            
            # 플립
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                x_norm = 1.0 - x_norm
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
                y_norm = 1.0 - y_norm
            
            # 색상 augmentation
            img = _apply_intensity_aug(img, strength=0.6)
            img = _apply_noise_blur(img, strength=0.4)
        
        img = np.clip(img, 0, 1)
        label_tensor = torch.tensor([x_norm, y_norm, r_norm], dtype=torch.float32)
        return torch.tensor(preprocess_image((img * 255).astype(np.uint8))), label_tensor


class MixedDataset(Dataset):
    """실제 + 합성 데이터 혼합 (오버샘플링 지원)"""
    def __init__(self, real_dataset, synthetic_dataset, real_ratio=0.9, oversample_real=4):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.real_ratio = real_ratio
        self.oversample_real = oversample_real
        self.total_len = len(real_dataset) * oversample_real

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if np.random.rand() < self.real_ratio:
            real_idx = idx % len(self.real_dataset)
            return self.real_dataset[real_idx]
        return self.synthetic_dataset[np.random.randint(len(self.synthetic_dataset))]
