"""
원형 시료 검출 모델 (CircleRegressor)

핵심 개선점:
1. Edge-aware Attention - 테두리 정보 강조
2. 반지름 중심 Loss - 반지름 예측 정확도 향상
3. 경량화된 ResNet 기반 구조
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_factory import SyntheticCircleDataset, IMG_SIZE


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    """Residual Block with optional stride"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x if self.skip is None else self.skip(x)
        return F.relu(out + identity)


class EdgeAttention(nn.Module):
    """Edge 채널 기반 Spatial Attention"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class CircleRegressor(nn.Module):
    """
    원형 시료 검출 회귀 모델
    
    입력: (batch, 6, H, W) - RGB(3) + Edge(1) + Coord(2)
    출력: (batch, 3) - [x_norm, y_norm, r_norm]
    """
    def __init__(self, in_channels=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = ResidualBlock(32, 48, stride=1)
        self.attention1 = EdgeAttention(48)
        self.layer2 = ResidualBlock(48, 96, stride=2)
        self.layer3 = ResidualBlock(96, 128, stride=2)
        self.attention2 = EdgeAttention(128)
        self.layer4 = ResidualBlock(128, 128, stride=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attention2(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.head(x)
        return torch.sigmoid(x)


def circle_iou_approx(pred, target):
    """
    원형 IoU 근사 계산 (MPS 호환, 빠름)
    
    중심 거리와 반지름 차이를 기반으로 IoU를 근사
    """
    px, py, pr = pred[:, 0], pred[:, 1], pred[:, 2]
    tx, ty, tr = target[:, 0], target[:, 1], target[:, 2]
    
    # 중심 거리 (normalized)
    d = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2 + 1e-8)
    
    # 반지름 유사도
    r_min = torch.min(pr, tr)
    r_max = torch.max(pr, tr) + 1e-8
    r_ratio = r_min / r_max
    
    # 거리 기반 겹침 비율 (원이 완전히 겹칠 때 1, 떨어질수록 0)
    overlap_ratio = torch.clamp(1.0 - d / (r_min + r_max + 1e-8), 0, 1)
    
    # IoU 근사: 반지름 유사도 * 겹침 비율
    iou_approx = r_ratio * overlap_ratio
    return iou_approx


def circle_loss(pred, target, xy_weight=1.5, r_weight=3.0, iou_weight=1.0):
    """
    원형 검출 손실 함수
    
    반지름에 더 높은 가중치를 부여하여 반지름 예측 정확도 향상
    IoU Loss 추가로 전체적인 원 매칭 품질 향상
    """
    # 위치 손실
    pred_xy = pred[:, :2].contiguous()
    target_xy = target[:, :2].contiguous()
    xy_loss = F.smooth_l1_loss(pred_xy, target_xy)
    
    # 반지름 손실 (중요!)
    r_loss = F.smooth_l1_loss(pred[:, 2].contiguous(), target[:, 2].contiguous())
    
    # IoU 손실 (근사)
    iou = circle_iou_approx(pred, target)
    iou_loss = 1.0 - iou.mean()
    
    return xy_weight * xy_loss + r_weight * r_loss + iou_weight * iou_loss


def train_synthetic(model_path, epochs, batch_size, lr, num_samples):
    """합성 데이터로 기본 모델 학습"""
    device = get_device()
    print(f"Device: {device}")
    
    dataset = SyntheticCircleDataset(num_samples=num_samples, domain_match=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = CircleRegressor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"기존 모델 로드: {model_path}")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = circle_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = running_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f} LR: {scheduler.get_last_lr()[0]:.6f}")
    
    torch.save(model.state_dict(), model_path)
    print(f"모델 저장: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="circle_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_synthetic(args.model, args.epochs, args.batch, args.lr, args.samples)
