# Day 2: 현미경 뷰 필드 원 검출 딥러닝 모델

## 목표
OpenCV 대신 딥러닝으로 현미경 이미지에서 뷰 필드 경계(원)를 검출하는 모델 개발

## 파일 구조

```
notebooks/day2/
├── data_factory.py      # 합성 데이터 생성
├── circle.py            # 기본 모델 학습
├── finetune.py          # 실제 이미지로 Fine-tuning
├── visualize_circle.py  # 결과 시각화 도구
├── labeling_tool.py     # 실제 이미지 라벨링 툴
├── train_all.sh         # 학습 자동화 스크립트
└── circle_model_v*.pth  # 저장된 모델들
```

## 모델 구조 (4층 CNN)

```python
model = nn.Sequential(
    # 1층: 256 -> 128
    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
    # 2층: 128 -> 64
    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
    # 3층: 64 -> 32
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
    # 4층: 32 -> 16
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
    
    nn.Flatten(),  # 16 * 16 * 128 = 32,768
    
    nn.Linear(128 * 16 * 16, 128),
    nn.LeakyReLU(0.1),  # Dead ReLU 방지
    nn.Linear(128, 3)   # 출력: x, y, r (정규화된 좌표)
)
```

## 학습 과정에서의 시행착오

### 1. AdaptiveAvgPool2d(1) 문제
- **증상**: Model Collapse (모든 이미지에 같은 위치 출력)
- **원인**: 공간 정보가 완전히 소멸
- **해결**: Flatten으로 교체

### 2. 5층 CNN (8x8) 문제
- **증상**: 여전히 Model Collapse, Loss 0.065에서 정체
- **원인**: 8x8은 너무 과한 압축, 원의 곡선 정보 소멸
- **해결**: 4층 CNN (16x16)으로 변경

### 3. ReLU 문제
- **증상**: 학습 초기 Dead ReLU 발생 가능
- **해결**: LeakyReLU(0.1)로 교체

### 4. 파라미터 폭발 문제
- **상황**: Flatten 후 Linear 층에서 6700만 개 파라미터
- **해결**: CNN 층을 늘려 feature map 크기 축소 후 Flatten

## 사용법

### 학습 실행
```bash
./notebooks/day2/train_all.sh
```
- 자동으로 버전 관리 (v1, v2, ...)
- 여러 터미널에서 병렬 학습 가능

### 이어서 학습
```bash
./notebooks/day2/train_all.sh  # 기존 모델에 덧바름
```

### 결과 확인
```bash
python notebooks/day2/visualize_circle.py
```
- 모델 선택 가능 (버전별)
- 합성 데이터 테스트 또는 실제 이미지 테스트

### 라벨링 (추가 데이터 필요시)
```bash
python notebooks/day2/labeling_tool.py /path/to/images
```
- 마우스 드래그로 원의 지름 지정
- `s`: 저장, `n`: 다음, `p`: 이전, `q`: 종료

## 현재 상태

- **v1 모델 학습 완료**
- 기본 학습 Loss: ~0.001
- Fine-tuning Loss: ~0.0006
- 총 학습 시간: ~8분

## TODO

- [ ] 더 많은 실제 이미지로 Fine-tuning
- [ ] 모델 정확도 검증
- [ ] 추론 속도 최적화
