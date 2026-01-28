"""
원형 검출 결과 시각화 모듈

사용법:
    # 합성 데이터 테스트
    python visualize_circle.py --mode synthetic --index 0
    
    # 실제 이미지 테스트
    python visualize_circle.py --mode image --image /path/to/image.jpg
    
    # labels.json 전체 검증
    python visualize_circle.py --mode validate --labels labels.json
"""

import argparse
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from circle import CircleRegressor, get_device
from data_factory import (
    SyntheticCircleDataset,
    preprocess_image,
    resize_with_padding,
    denormalize_from_square,
    normalize_label_to_square,
    IMG_SIZE,
)


def _load_model(model_path):
    """모델 로드"""
    device = get_device()
    model = CircleRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _predict_on_tensor(model, img_tensor):
    """텐서 입력으로 예측"""
    device = get_device()
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    return pred


def visualize_synthetic(model_path, index):
    """합성 데이터 시각화"""
    model = _load_model(model_path)
    dataset = SyntheticCircleDataset(num_samples=max(index + 1, 20))
    img_tensor, label = dataset[index]
    pred = _predict_on_tensor(model, img_tensor)
    
    img = img_tensor[:3].permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8).copy()
    
    true_x, true_y, true_r = (label.numpy() * IMG_SIZE).astype(int)
    pred_x, pred_y, pred_r = (pred * IMG_SIZE).astype(int)
    
    cv2.circle(img, (true_x, true_y), true_r, (0, 255, 0), 2)
    cv2.circle(img, (pred_x, pred_y), pred_r, (255, 0, 0), 2)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"GT(초록) x={true_x}, y={true_y}, r={true_r}\nPred(빨강) x={pred_x}, y={pred_y}, r={pred_r}")
    plt.tight_layout()
    plt.show()


def visualize_image(model_path, image_path, gt_label=None):
    """실제 이미지 시각화"""
    model = _load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    img_resized, meta = resize_with_padding(img, IMG_SIZE)
    img_tensor = torch.tensor(preprocess_image(img_resized))
    pred = _predict_on_tensor(model, img_tensor)
    
    pred_x, pred_y, pred_r = denormalize_from_square(pred, meta, IMG_SIZE)
    
    display = img.copy()
    
    # GT가 있으면 표시
    if gt_label:
        gt_x = int(gt_label[0] * w)
        gt_y = int(gt_label[1] * h)
        gt_r = int(gt_label[2] * min(w, h))
        cv2.circle(display, (gt_x, gt_y), gt_r, (0, 255, 0), 3)
        cv2.circle(display, (gt_x, gt_y), 5, (0, 255, 0), -1)
    
    # 예측 표시
    cv2.circle(display, (pred_x, pred_y), pred_r, (255, 0, 0), 3)
    cv2.circle(display, (pred_x, pred_y), 5, (255, 0, 0), -1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(display)
    plt.axis("off")
    title = f"{os.path.basename(image_path)}\nPred: x={pred_x}, y={pred_y}, r={pred_r}"
    if gt_label:
        title += f"\nGT: x={gt_x}, y={gt_y}, r={gt_r}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def validate_labels(model_path, labels_file, show_worst=5):
    """labels.json 전체 검증"""
    model = _load_model(model_path)
    
    with open(labels_file, "r") as f:
        labels = json.load(f)
    
    errors = []
    
    for img_path, info in labels.items():
        if not os.path.exists(img_path):
            print(f"Skip (not found): {img_path}")
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        img_resized, meta = resize_with_padding(img, IMG_SIZE)
        img_tensor = torch.tensor(preprocess_image(img_resized))
        pred = _predict_on_tensor(model, img_tensor)
        
        # GT 좌표
        if "x_norm" in info:
            gt_x_norm, gt_y_norm, gt_r_norm = info["x_norm"], info["y_norm"], info["r_norm"]
        else:
            gt_x_norm = info["x"] / info["width"]
            gt_y_norm = info["y"] / info["height"]
            gt_r_norm = info["r"] / min(info["width"], info["height"])
        
        # 정사각형 좌표로 변환
        gt_x_sq, gt_y_sq, gt_r_sq = normalize_label_to_square(gt_x_norm, gt_y_norm, gt_r_norm, meta, IMG_SIZE)
        
        # 에러 계산
        xy_error = np.sqrt((pred[0] - gt_x_sq) ** 2 + (pred[1] - gt_y_sq) ** 2)
        r_error = abs(pred[2] - gt_r_sq)
        total_error = xy_error + r_error * 2  # 반지름 에러에 가중치
        
        errors.append({
            "path": img_path,
            "pred": pred,
            "gt": [gt_x_sq, gt_y_sq, gt_r_sq],
            "xy_error": xy_error,
            "r_error": r_error,
            "total_error": total_error,
        })
    
    # 통계
    xy_errors = [e["xy_error"] for e in errors]
    r_errors = [e["r_error"] for e in errors]
    
    print(f"\n=== 검증 결과 ({len(errors)}개 이미지) ===")
    print(f"XY Error: mean={np.mean(xy_errors):.4f}, std={np.std(xy_errors):.4f}")
    print(f"R Error: mean={np.mean(r_errors):.4f}, std={np.std(r_errors):.4f}")
    
    # 최악의 케이스 시각화
    errors.sort(key=lambda x: x["total_error"], reverse=True)
    print(f"\n최악의 {show_worst}개 케이스:")
    for i, e in enumerate(errors[:show_worst]):
        print(f"  {i+1}. {os.path.basename(e['path'])}: XY={e['xy_error']:.4f}, R={e['r_error']:.4f}")
    
    # 최악의 케이스 시각화
    if show_worst > 0:
        fig, axes = plt.subplots(1, min(show_worst, len(errors)), figsize=(5 * min(show_worst, len(errors)), 5))
        if show_worst == 1:
            axes = [axes]
        
        for ax, e in zip(axes, errors[:show_worst]):
            img = cv2.imread(e["path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized, meta = resize_with_padding(img, IMG_SIZE)
            
            gt = e["gt"]
            pred = e["pred"]
            
            display = img_resized.copy()
            gt_x, gt_y, gt_r = int(gt[0] * IMG_SIZE), int(gt[1] * IMG_SIZE), int(gt[2] * IMG_SIZE)
            pred_x, pred_y, pred_r = int(pred[0] * IMG_SIZE), int(pred[1] * IMG_SIZE), int(pred[2] * IMG_SIZE)
            
            cv2.circle(display, (gt_x, gt_y), gt_r, (0, 255, 0), 2)
            cv2.circle(display, (pred_x, pred_y), pred_r, (255, 0, 0), 2)
            
            ax.imshow(display)
            ax.axis("off")
            ax.set_title(f"R_err: {e['r_error']:.3f}")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="circle_model_finetuned_best.pth")
    parser.add_argument("--mode", choices=["synthetic", "image", "validate"], default="image")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image", default="")
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--show-worst", type=int, default=5)
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.mode == "synthetic":
        visualize_synthetic(args.model, args.index)
    elif args.mode == "validate":
        validate_labels(args.model, args.labels, args.show_worst)
    else:
        visualize_image(args.model, args.image)
