import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class CircleDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.images = []
        self.labels = []

        for _ in range(num_samples):
            # 1. 배경색 랜덤 설정 (어두운 톤부터 밝은 톤까지)
            bg_color = np.random.rand(3).astype(np.float32)
            img = np.full((128, 128, 3), bg_color, dtype=np.float32)
            
            # 2. 랜덤 좌표 및 반지름 생성 (이미지 범위 안에서)
            r = np.random.randint(10, 63) # 반지름 10~63
            
            x = np.random.randint(r, 128 - r) # x좌표 + 반지름이 128보다 크면 안됨
            y = np.random.randint(r, 128 - r) # y좌표 + 반지름이 128보다 크면 안됨
            
            # 3. 원의 랜덤 색상 (R, G, B 각각 0.3~1.0 사이)
            circle_color = []
            for i in range(3): # 배경색이 밝으면 원을 어둡게, 배경이 어두우면 원을 밝게
                if bg_color[i] > 0.5:
                    circle_color.append(np.random.uniform(0.0, 0.3)) # 어두운 색
                else:
                    circle_color.append(np.random.uniform(0.7, 1.0)) # 밝은 색
            
            # 4. 원 그리기
            cv2.circle(img, (x, y), r, tuple(circle_color), -1)
            
            # 5. 약간의 노이즈 추가 (현미경 특유의 거친 느낌)
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
            
            self.images.append(img.transpose(2, 0, 1))
            self.labels.append(np.array([x/128, y/128, r/128], dtype=np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])

# 공장 가동! 5000개의 데이터 생성 (정확도 향상을 위한 데이터 증설)
if __name__ == "__main__":
    dataset = CircleDataset(num_samples=5000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"공장 가동 완료! 생성된 데이터 셋: {len(dataset)}개")