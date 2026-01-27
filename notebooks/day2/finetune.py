"""
실제 이미지로 Fine-tuning
- 라벨링된 실제 현미경 이미지로 모델 미세조정
- 기존 학습된 모델 가중치 로드 후 추가 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import os

IMG_SIZE = 256

def create_model():
    """circle.py와 동일한 경량 모델 구조"""
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
    return model


class RealImageDataset(Dataset):
    """라벨링된 실제 이미지 데이터셋 - 미리 로드"""
    def __init__(self, labels_file, augment=True):
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        self.image_paths = list(self.labels.keys())
        self.augment = augment
        
        # 미리 모든 이미지 로드 및 전처리
        print(f"이미지 로드 중... ({len(self.image_paths)}개)")
        self.images = []
        self.label_tensors = []
        
        for i, img_path in enumerate(self.image_paths):
            print(f"  [{i+1}/{len(self.image_paths)}] {os.path.basename(img_path)}")
            
            label_data = self.labels[img_path]
            
            # 이미지 로드
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # 256x256로 리사이즈
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # 좌표 스케일 변환
            x = label_data['x']
            y = label_data['y']
            r = label_data['r']
            
            scale_x = IMG_SIZE / w
            scale_y = IMG_SIZE / h
            scale_r = min(scale_x, scale_y)
            
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            r_scaled = r * scale_r
            
            # 정규화 (0~1)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            self.images.append(img_normalized)
            self.label_tensors.append([
                x_scaled / IMG_SIZE,
                y_scaled / IMG_SIZE,
                r_scaled / IMG_SIZE
            ])
        
        print(f"로드 완료!")
    
    def __len__(self):
        # Augmentation으로 데이터 증폭
        return len(self.image_paths) * (10 if self.augment else 1)
    
    def __getitem__(self, idx):
        # 원본 인덱스
        real_idx = idx % len(self.image_paths)
        img = self.images[real_idx].copy()
        label = self.label_tensors[real_idx].copy()
        
        # Augmentation
        if self.augment and idx >= len(self.image_paths):
            # 밝기 변화
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 1)
            
            # 노이즈
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
        
        img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img_tensor, label_tensor


def finetune(labels_file, pretrained_model_path, output_path, epochs=100, lr=0.0001):
    """Fine-tuning 실행"""
    print("=== Fine-tuning 시작 ===\n")
    
    # 데이터셋 로드
    dataset = RealImageDataset(labels_file, augment=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 모델 로드
    model = create_model()
    
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print(f"사전학습 모델 로드: {pretrained_model_path}")
    else:
        print("사전학습 모델 없음, 처음부터 학습")
    
    # Fine-tuning 설정 (낮은 학습률)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
    print(f"원본 데이터: {len(dataset.image_paths)}개")
    print(f"Augmentation 후: {len(dataset)}개")
    print(f"에포크: {epochs}회")
    print(f"학습률: {lr}")
    print()
    
    # 학습
    print("\n학습 시작...")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(loader)
        
        # 매 에포크마다 진행 표시
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # 저장
    torch.save(model.state_dict(), output_path)
    print(f"\n=== Fine-tuning 완료 ===")
    print(f"모델 저장: {output_path}")
    
    return model


if __name__ == "__main__":
    import sys
    
    # 경로 설정
    if len(sys.argv) > 1:
        labels_file = sys.argv[1]
    else:
        labels_file = "/Users/sjlee/Downloads/스캔슬라이드1/labels.json"
    
    pretrained_model = "circle_model.pth"
    output_model = "circle_model_finetuned.pth"
    
    if not os.path.exists(labels_file):
        print(f"라벨 파일이 없습니다: {labels_file}")
        print("먼저 labeling_tool.py로 라벨링을 해주세요:")
        print(f"  python notebooks/day2/labeling_tool.py")
    else:
        finetune(labels_file, pretrained_model, output_model, epochs=100, lr=0.0001)
