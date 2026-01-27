"""
현미경 이미지 라벨링 도구
- 마우스 클릭: 원의 중심 지정
- 드래그: 반지름 지정
- 's' 키: 저장 후 다음 이미지
- 'r' 키: 현재 라벨 초기화
- 'q' 키: 종료
"""

import cv2
import numpy as np
import os
import json
from glob import glob

class LabelingTool:
    def __init__(self, image_dir, output_file='labels.json'):
        self.image_dir = image_dir
        self.output_file = os.path.join(image_dir, output_file)
        
        # 이미지 파일 목록
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            self.image_files.extend(glob(os.path.join(image_dir, '**', ext), recursive=True))
        self.image_files = sorted(self.image_files)
        
        # 기존 라벨 로드
        self.labels = {}
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                self.labels = json.load(f)
        
        self.current_idx = 0
        self.center = None
        self.radius = 0
        self.dragging = False
        self.start_point = None  # 지름의 시작점
        self.end_point = None    # 지름의 끝점
        self.img = None
        self.img_display = None
        self.scale = 1.0
        
    def mouse_callback(self, event, x, y, flags, param):
        # 스케일 보정
        real_x = int(x / self.scale)
        real_y = int(y / self.scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 지름의 시작점
            self.start_point = (real_x, real_y)
            self.end_point = None
            self.center = None
            self.radius = 0
            self.dragging = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.start_point:
                self.end_point = (real_x, real_y)
                # 중심 = 두 점의 중간
                self.center = (
                    (self.start_point[0] + self.end_point[0]) // 2,
                    (self.start_point[1] + self.end_point[1]) // 2
                )
                # 반지름 = 거리 / 2
                dx = self.end_point[0] - self.start_point[0]
                dy = self.end_point[1] - self.start_point[1]
                self.radius = int(np.sqrt(dx*dx + dy*dy) / 2)
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            if self.center and self.radius > 10:
                print(f"  라벨: center=({self.center[0]}, {self.center[1]}), radius={self.radius}, diameter={self.radius*2}")
    
    def update_display(self):
        self.img_display = self.img.copy()
        
        if self.center and self.radius > 0:
            # 원 그리기 (빨간색)
            cv2.circle(self.img_display, self.center, self.radius, (0, 0, 255), 3)
            # 중심점 (초록색)
            cv2.circle(self.img_display, self.center, 5, (0, 255, 0), -1)
            # 지름 선 (파란색) - 드래그 시각화
            if self.start_point and self.end_point:
                cv2.line(self.img_display, self.start_point, self.end_point, (255, 0, 0), 2)
                cv2.circle(self.img_display, self.start_point, 5, (255, 0, 0), -1)
                cv2.circle(self.img_display, self.end_point, 5, (255, 0, 0), -1)
        
        # 정보 표시
        h, w = self.img_display.shape[:2]
        info = f"[{self.current_idx+1}/{len(self.image_files)}] "
        if self.center:
            info += f"center=({self.center[0]}, {self.center[1]}), d={self.radius*2}"
        cv2.putText(self.img_display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(self.img_display, "Drag diameter | s:save  r:reset  q:quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 리사이즈해서 표시
        display = cv2.resize(self.img_display, None, fx=self.scale, fy=self.scale)
        cv2.imshow('Labeling Tool', display)
    
    def save_label(self):
        if self.center and self.radius > 10:
            img_path = self.image_files[self.current_idx]
            h, w = self.img.shape[:2]
            
            # 정규화된 좌표로 저장
            self.labels[img_path] = {
                'x': self.center[0],
                'y': self.center[1],
                'r': self.radius,
                'width': w,
                'height': h,
                # 정규화 값 (0~1)
                'x_norm': self.center[0] / w,
                'y_norm': self.center[1] / h,
                'r_norm': self.radius / min(w, h)
            }
            
            # 파일에 저장
            with open(self.output_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            print(f"  저장 완료: {os.path.basename(img_path)}")
            return True
        else:
            print("  라벨이 없거나 반지름이 너무 작습니다.")
            return False
    
    def run(self):
        print(f"\n=== 라벨링 도구 시작 ===")
        print(f"이미지 경로: {self.image_dir}")
        print(f"총 이미지: {len(self.image_files)}개")
        print(f"기존 라벨: {len(self.labels)}개")
        print()
        print("조작법:")
        print("  - 마우스 클릭 + 드래그: 원 그리기")
        print("  - 's': 저장 후 다음 이미지")
        print("  - 'r': 현재 라벨 초기화")
        print("  - 'n': 다음 이미지 (저장 안함)")
        print("  - 'p': 이전 이미지")
        print("  - 'q': 종료")
        print()
        
        cv2.namedWindow('Labeling Tool')
        cv2.setMouseCallback('Labeling Tool', self.mouse_callback)
        
        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            print(f"\n[{self.current_idx+1}/{len(self.image_files)}] {os.path.basename(img_path)}")
            
            # 이미지 로드
            self.img = cv2.imread(img_path)
            if self.img is None:
                print(f"  이미지 로드 실패")
                self.current_idx += 1
                continue
            
            h, w = self.img.shape[:2]
            print(f"  크기: {w} x {h}")
            
            # 화면에 맞게 스케일 조정
            max_display = 1000
            self.scale = min(max_display / w, max_display / h, 1.0)
            
            # 기존 라벨이 있으면 로드
            if img_path in self.labels:
                label = self.labels[img_path]
                self.center = (label['x'], label['y'])
                self.radius = label['r']
                print(f"  기존 라벨 로드: x={self.center[0]}, y={self.center[1]}, r={self.radius}")
            else:
                self.center = None
                self.radius = 0
            
            self.update_display()
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 저장 후 다음
                    if self.save_label():
                        self.current_idx += 1
                        break
                        
                elif key == ord('r'):  # 초기화
                    self.center = None
                    self.radius = 0
                    self.update_display()
                    print("  라벨 초기화")
                    
                elif key == ord('n'):  # 다음 (저장 안함)
                    self.current_idx += 1
                    break
                    
                elif key == ord('p'):  # 이전
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        break
                        
                elif key == ord('q'):  # 종료
                    print(f"\n=== 종료 ===")
                    print(f"총 라벨: {len(self.labels)}개")
                    print(f"저장 위치: {self.output_file}")
                    cv2.destroyAllWindows()
                    return
        
        print(f"\n=== 모든 이미지 완료 ===")
        print(f"총 라벨: {len(self.labels)}개")
        print(f"저장 위치: {self.output_file}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    # 이미지 경로 설정
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = "/Users/sjlee/Downloads/스캔슬라이드1"
    
    tool = LabelingTool(image_dir)
    tool.run()
