import torch
import torch.nn as nn
from data_factory import CircleDataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
import numpy as np

def create_model():
    """circle.py와 동일한 모델 구조 생성"""
    model = nn.Sequential(
        # 특징 추출 (CNN Layer)
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 128 -> 64
        
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 64 -> 32
        
        # 지능형 요약 (Flatten)
        nn.Flatten(), 
        
        # 좌표 추출 (Regression Head)
        nn.Linear(32 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 3) # 최종 출력: x, y, r
    )
    return model

class CircleVisualizer:
    """인터랙티브 원 예측 시각화 도구"""
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.current_index = 0
        self.max_index = len(dataset) - 1
        
        # Figure와 Axes 설정
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        plt.subplots_adjust(bottom=0.15)
        
        # 버튼 위치 설정
        ax_prev = plt.axes([0.2, 0.05, 0.1, 0.05])
        ax_next = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_first = plt.axes([0.05, 0.05, 0.1, 0.05])
        ax_last = plt.axes([0.85, 0.05, 0.1, 0.05])
        
        # 버튼 생성
        self.btn_first = Button(ax_first, '<<')
        self.btn_prev = Button(ax_prev, '<')
        self.btn_next = Button(ax_next, '>')
        self.btn_last = Button(ax_last, '>>')
        
        # 버튼 이벤트 연결
        self.btn_first.on_clicked(self.goto_first)
        self.btn_prev.on_clicked(self.goto_prev)
        self.btn_next.on_clicked(self.goto_next)
        self.btn_last.on_clicked(self.goto_last)
        
        # 초기 이미지 표시
        self.update_image()
        
    def update_image(self):
        """현재 인덱스의 이미지 업데이트"""
        self.ax.clear()
        
        # 1. 데이터 추출
        img_tensor, label = self.dataset[self.current_index]
        
        # 2. 모델 예측
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(img_tensor.unsqueeze(0))
            prediction = prediction.squeeze().numpy()
        
        # 3. 텐서를 이미지로 변환
        display_img = img_tensor.permute(1, 2, 0).numpy()
        display_img = (display_img * 255).astype(np.uint8).copy()
        
        # 4. 정규화된 값 복원 (0~1 -> 0~128)
        true_x, true_y, true_r = (label * 128).numpy().astype(int)
        pred_x, pred_y, pred_r = (prediction * 128).astype(int)
        
        # 5. 오차 계산
        error_x = abs(true_x - pred_x)
        error_y = abs(true_y - pred_y)
        error_r = abs(true_r - pred_r)
        
        # 6. 원 그리기
        cv2.circle(display_img, (true_x, true_y), true_r, (0, 255, 0), 2)
        cv2.circle(display_img, (pred_x, pred_y), pred_r, (255, 0, 0), 2)
        
        # 7. 이미지 표시
        self.ax.imshow(display_img)
        self.ax.axis('off')
        
        # 8. 제목 및 정보 표시
        title = f"Index: {self.current_index}/{self.max_index}\n\n"
        title += f"정답 (초록): x={true_x}, y={true_y}, r={true_r}\n"
        title += f"예측 (빨강): x={pred_x}, y={pred_y}, r={pred_r}\n"
        title += f"오차: Δx={error_x}, Δy={error_y}, Δr={error_r}"
        
        self.ax.set_title(title, fontsize=11, pad=10)
        
        self.fig.canvas.draw()
    
    def goto_first(self, event):
        """첫 번째 이미지로 이동"""
        self.current_index = 0
        self.update_image()
    
    def goto_prev(self, event):
        """이전 이미지로 이동"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()
    
    def goto_next(self, event):
        """다음 이미지로 이동"""
        if self.current_index < self.max_index:
            self.current_index += 1
            self.update_image()
    
    def goto_last(self, event):
        """마지막 이미지로 이동"""
        self.current_index = self.max_index
        self.update_image()
    
    def show(self):
        """GUI 표시"""
        plt.show()

# 메인 실행 코드
if __name__ == "__main__":
    # 1. 데이터셋 생성 (시각화용은 100개면 충분)
    dataset = CircleDataset(num_samples=100)
    
    # 2. 모델 생성 및 저장된 가중치 로드
    model = create_model()
    model.load_state_dict(torch.load('circle_model.pth'))
    model.eval()
    
    print("--- 모델 로드 완료! ---")
    print(f"데이터셋 크기: {len(dataset)}개")
    print("GUI 조작법:")
    print("  << : 첫 번째 이미지")
    print("  <  : 이전 이미지")
    print("  >  : 다음 이미지")
    print("  >> : 마지막 이미지")
    print()
    
    # 3. 인터랙티브 시각화 시작
    visualizer = CircleVisualizer(dataset, model)
    visualizer.show()
