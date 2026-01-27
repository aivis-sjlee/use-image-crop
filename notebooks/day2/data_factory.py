import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 256  # 해상도 설정

class CircleDataset(Dataset):
    """현미경 뷰 필드 스타일 데이터셋 (256x256) - 개선 버전
    - 다양한 augmentation 적용
    - 원이 잘리는 케이스 포함
    - 더 현실적인 경계 처리
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.images = []
        self.labels = []
        
        print(f"데이터 생성 중... ({num_samples}개)")
        for i in range(num_samples):
            if (i + 1) % 500 == 0:
                print(f"  진행: {i + 1}/{num_samples}")
            # 1. 배경: 검은색~진한 회색 (더 다양하게)
            bg_brightness = np.random.uniform(0.0, 0.2)
            # 약간의 색조 변화
            bg_color = np.array([
                bg_brightness * np.random.uniform(0.8, 1.2),
                bg_brightness * np.random.uniform(0.8, 1.2),
                bg_brightness * np.random.uniform(0.9, 1.1)
            ], dtype=np.float32)
            bg_color = np.clip(bg_color, 0, 0.25)
            img = np.full((IMG_SIZE, IMG_SIZE, 3), bg_color, dtype=np.float32)
            
            # 2. 원 위치: 더 넓은 범위 (원이 잘리는 케이스 포함)
            x = np.random.randint(80, 176)  # 더 넓은 범위
            y = np.random.randint(80, 176)
            
            # 3. 반지름: 더 다양하게 (80~150)
            r = np.random.randint(80, 150)
            
            # 4. 뷰 필드 내부 색상: 더 다양한 밝기
            brightness_factor = np.random.uniform(0.7, 1.0)
            base_color = np.array([
                np.random.uniform(0.80, 0.98) * brightness_factor,
                np.random.uniform(0.75, 0.95) * brightness_factor,
                np.random.uniform(0.70, 0.92) * brightness_factor,
            ], dtype=np.float32)
            
            # 5. 원 그리기 (뷰 필드 기본)
            cv2.circle(img, (x, y), r, (float(base_color[0]), float(base_color[1]), float(base_color[2])), -1)
            
            # 6. 원 내부에 세포 같은 텍스처 추가 (속도를 위해 줄임)
            num_dots = np.random.randint(20, 50)
            for _ in range(num_dots):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, r * 0.95)
                dot_x = int(x + dist * np.cos(angle))
                dot_y = int(y + dist * np.sin(angle))
                
                if 0 <= dot_x < IMG_SIZE and 0 <= dot_y < IMG_SIZE:
                    dot_size = np.random.randint(1, 8)
                    # 다양한 색상 (파란색, 보라색, 갈색)
                    color_type = np.random.choice(['blue', 'brown', 'dark'])
                    if color_type == 'blue':
                        dot_color = (np.random.uniform(0.2, 0.5), np.random.uniform(0.3, 0.6), np.random.uniform(0.5, 0.8))
                    elif color_type == 'brown':
                        dot_color = (np.random.uniform(0.4, 0.7), np.random.uniform(0.2, 0.5), np.random.uniform(0.1, 0.3))
                    else:
                        dot_color = (np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4))
                    cv2.circle(img, (dot_x, dot_y), dot_size, dot_color, -1)
            
            # 7. 경계 부드럽게 (더 강한 블러 - 현실적)
            blur_strength = np.random.choice([0, 3, 5, 7], p=[0.2, 0.3, 0.3, 0.2])
            if blur_strength > 0:
                img = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
            
            # 8. 전체 노이즈 추가
            noise_level = np.random.uniform(0.01, 0.04)
            noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
            
            # 9. 밝기/대비 augmentation
            if np.random.rand() < 0.5:
                brightness_shift = np.random.uniform(-0.1, 0.1)
                contrast = np.random.uniform(0.8, 1.2)
                img = np.clip((img - 0.5) * contrast + 0.5 + brightness_shift, 0, 1)
            
            self.images.append(img.transpose(2, 0, 1).astype(np.float32))
            self.labels.append(np.array([x/IMG_SIZE, y/IMG_SIZE, r/IMG_SIZE], dtype=np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])

# 공장 가동! 5000개의 데이터 생성 (정확도 향상을 위한 데이터 증설)
if __name__ == "__main__":
    dataset = CircleDataset(num_samples=5000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"공장 가동 완료! 생성된 데이터 셋: {len(dataset)}개")