import torch
import torch.nn as nn
from data_factory import CircleDataset, IMG_SIZE
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import cv2
import numpy as np
import os

def create_model():
    """circle.py와 동일한 경량 모델 구조"""
    model = nn.Sequential(
        # 1층: 256 -> 128
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        # 2층: 128 -> 64
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        # 3층: 64 -> 32
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        # Global Average Pooling
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        
        # Regression Head
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
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
        
        # 4. 정규화된 값 복원 (0~1 -> 0~IMG_SIZE)
        true_x, true_y, true_r = (label * IMG_SIZE).numpy().astype(int)
        pred_x, pred_y, pred_r = (prediction * IMG_SIZE).astype(int)
        
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


class ImageTester:
    """실제 이미지 테스트 도구 (터미널에서 경로 입력)"""
    def __init__(self, model, image_path):
        self.model = model
        self.image_path = image_path
        
    def run(self):
        """이미지 로드 및 예측 실행"""
        # 이미지 로드
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"이미지 로드 실패: {self.image_path}")
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]  # (H, W)
        
        print(f"이미지 로드 완료: {self.image_path}")
        print(f"원본 크기: {original_size[1]} x {original_size[0]}")
        
        # 모델 입력용: IMG_SIZExIMG_SIZE로 리사이즈 + 정규화
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_normalized.transpose(2, 0, 1))
        
        # 모델 예측
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(img_tensor.unsqueeze(0))
            prediction = prediction.squeeze().numpy()
        
        print(f"모델 예측값 (정규화): {prediction}")
        
        # 예측값 복원 (0~1 -> 실제 픽셀 좌표)
        pred_x_scaled = int(prediction[0] * IMG_SIZE)
        pred_y_scaled = int(prediction[1] * IMG_SIZE)
        pred_r_scaled = int(prediction[2] * IMG_SIZE)
        
        # 원본 이미지 크기로 스케일 변환
        scale_x = original_size[1] / IMG_SIZE
        scale_y = original_size[0] / IMG_SIZE
        scale_r = min(scale_x, scale_y)
        
        pred_x = int(pred_x_scaled * scale_x)
        pred_y = int(pred_y_scaled * scale_y)
        pred_r = int(pred_r_scaled * scale_r)
        
        print(f"예측 원 (빨강): x={pred_x}, y={pred_y}, r={pred_r}")
        
        # 원본 이미지에 예측 원 그리기 (빨간색)
        display_img = img.copy()
        cv2.circle(display_img, (pred_x, pred_y), pred_r, (255, 0, 0), 3)
        cv2.circle(display_img, (pred_x, pred_y), 5, (255, 0, 0), -1)
        
        # 화면 표시
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(display_img)
        ax.axis('off')
        
        title = f"Image: {os.path.basename(self.image_path)}\n"
        title += f"Size: {original_size[1]} x {original_size[0]}\n\n"
        title += f"Predicted Circle (Red): x={pred_x}, y={pred_y}, r={pred_r}"
        
        ax.set_title(title, fontsize=12, pad=10)
        plt.tight_layout()
        plt.show()


# 메인 실행 코드
if __name__ == "__main__":
    # 모델 로드
    model = create_model()
    model.load_state_dict(torch.load('circle_model.pth'))
    model.eval()
    print("--- 모델 로드 완료! ---\n")
    
    # 모드 선택
    print("=== 모드 선택 ===")
    print("1. 생성된 데이터셋으로 테스트 (정답 비교)")
    print("2. 실제 이미지 업로드 테스트 (병리 현미경 등)")
    print()
    
    mode = input("모드 선택 (1 or 2): ").strip()
    
    if mode == "2":
        # 이미지 테스트 모드
        print("\n--- 이미지 테스트 모드 ---")
        print("테스트할 이미지 경로를 입력하세요.")
        print("(Finder에서 파일을 터미널로 드래그하면 경로가 자동 입력됩니다)\n")
        
        image_path = input("이미지 경로: ").strip().strip("'\"")  # 따옴표 제거
        
        if not os.path.exists(image_path):
            print(f"파일이 존재하지 않습니다: {image_path}")
        else:
            tester = ImageTester(model, image_path)
            tester.run()
    else:
        # 데이터셋 테스트 모드
        dataset = CircleDataset(num_samples=100)
        print(f"\n--- 데이터셋 테스트 모드 ---")
        print(f"데이터셋 크기: {len(dataset)}개")
        print("GUI 조작법:")
        print("  << : 첫 번째 이미지")
        print("  <  : 이전 이미지")
        print("  >  : 다음 이미지")
        print("  >> : 마지막 이미지")
        print()
        
        visualizer = CircleVisualizer(dataset, model)
        visualizer.show()
