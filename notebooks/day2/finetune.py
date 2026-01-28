"""
실제 이미지 파인튜닝 모듈

핵심 개선점:
1. 실제 데이터 집중 학습 (95% 비율)
2. 오버샘플링으로 데이터 부족 보완
3. 긴 patience + warm restart
4. 더 강력한 augmentation
"""

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from circle import CircleRegressor, circle_loss, get_device
from data_factory import RealImageDataset, SyntheticCircleDataset, MixedDataset, IMG_SIZE


def _split_indices(n, val_ratio=0.15, seed=42):
    """Train/Val 분리 (실제 데이터가 적으므로 val 비율 낮춤)"""
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = max(2, int(n * val_ratio))
    return indices[val_size:], indices[:val_size]


def _evaluate(model, loader, device):
    """검증 손실 계산"""
    model.eval()
    total_loss = 0.0
    total_r_error = 0.0
    count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += circle_loss(outputs, labels).item()
            total_r_error += torch.abs(outputs[:, 2] - labels[:, 2]).sum().item()
            count += inputs.size(0)
    return total_loss / len(loader), total_r_error / count


def finetune(labels_file, pretrained_model_path, output_path, epochs, lr, real_ratio, batch_size):
    """실제 이미지 파인튜닝"""
    device = get_device()
    print(f"Device: {device}")
    print(f"Labels: {labels_file}")
    print(f"Pretrained: {pretrained_model_path}")
    print(f"Output: {output_path}")
    
    # 데이터셋 준비
    real_dataset = RealImageDataset(labels_file, augment=True)
    train_idx, val_idx = _split_indices(len(real_dataset), val_ratio=0.15, seed=42)
    real_train = Subset(real_dataset, train_idx)
    real_val = Subset(RealImageDataset(labels_file, augment=False), val_idx)
    
    print(f"실제 데이터: {len(real_dataset)}개 (Train: {len(train_idx)}, Val: {len(val_idx)})")
    
    # Mixed Dataset (실제 데이터 오버샘플링)
    if real_ratio < 1.0:
        synthetic_dataset = SyntheticCircleDataset(num_samples=max(500, len(real_dataset)), domain_match=True)
        train_dataset = MixedDataset(real_train, synthetic_dataset, real_ratio=real_ratio, oversample_real=6)
    else:
        # 실제 데이터만 사용 시 강제 오버샘플링
        train_dataset = MixedDataset(real_train, SyntheticCircleDataset(100), real_ratio=1.0, oversample_real=8)
    
    print(f"학습 데이터: {len(train_dataset)}개 (Real ratio: {real_ratio})")
    
    # DataLoader
    num_workers = 0 if torch.backends.mps.is_available() else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(real_val, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 모델 로드
    model = CircleRegressor().to(device)
    
    loaded_from = None
    if os.path.exists(output_path):
        model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
        loaded_from = output_path
    elif os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device, weights_only=True))
        loaded_from = pretrained_model_path
    
    if loaded_from:
        print(f"모델 로드: {loaded_from}")
    else:
        print("새 모델 초기화")
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # 학습
    best_val = float("inf")
    best_r_error = float("inf")
    patience = 15
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = circle_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        
        train_loss = running / len(train_loader)
        val_loss, r_error = _evaluate(model, val_loader, device)
        scheduler.step()
        
        # 로깅
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss:.5f} Val: {val_loss:.5f} R_err: {r_error:.4f}")
        
        # Best 모델 저장 (반지름 에러 기준)
        improved = False
        if r_error < best_r_error:
            best_r_error = r_error
            improved = True
        if val_loss < best_val:
            best_val = val_loss
            improved = True
        
        if improved:
            wait = 0
            torch.save(model.state_dict(), output_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 최종 모델 로드
    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
    
    # 최종 검증
    final_loss, final_r_error = _evaluate(model, val_loader, device)
    print(f"\n=== 학습 완료 ===")
    print(f"모델: {output_path}")
    print(f"최종 Val Loss: {final_loss:.5f}")
    print(f"최종 R Error: {final_r_error:.4f}")
    print(f"이미지 크기: {IMG_SIZE}x{IMG_SIZE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--base", default="circle_model.pth")
    parser.add_argument("--out", default="circle_model_finetuned_best.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--real-ratio", type=float, default=0.95)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    finetune(args.labels, args.base, args.out, args.epochs, args.lr, args.real_ratio, args.batch)
