# Day 1: PyTorch 신경망 기초

## 1. PyTorch 텐서 기초

```python
import torch

# 랜덤 텐서 생성
x = torch.rand(3, 3)

# 텐서 속성 확인
x.shape  # torch.Size([3, 3])
x.dtype  # torch.float32

# 직접 값 지정하여 텐서 생성
image = torch.tensor([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6],
                      [0.7, 0.8, 0.9]])
```

---

## 2. 뉴런의 기본 연산

### 핵심 구성 요소
| 요소 | 설명 |
|------|------|
| **가중치 (w)** | 각 입력의 중요도를 결정하는 학습 파라미터 |
| **편향 (b)** | 결과값을 조정하는 상수 |
| **활성화 함수** | 비선형성을 부여하여 복잡한 패턴 학습 가능하게 함 |

### 기본 수식
```
z = Σ(input × weight) + bias
output = activation(z)
```

### 코드 예시
```python
w = torch.randn(3, 3, requires_grad=True)  # 학습 가능한 가중치
b = torch.randn(1, requires_grad=True)     # 학습 가능한 편향

z = torch.sum(image * w) + b  # 선형 연산
a = torch.relu(z)             # ReLU 활성화 (음수 → 0)
```

---

## 3. 이진 분류 모델 구현

### 문제 설정: "이 이미지가 갈색인가?"

**갈색 이미지 (정답: 1)**
```python
brown_image = torch.tensor([
    [[0.1, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.1]],  # R (중앙 높음)
    [[0.8, 0.8, 0.8], [0.8, 0.3, 0.8], [0.8, 0.8, 0.8]],  # G (중앙 낮음)
    [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]   # B
])
```

**초록 이미지 (정답: 0)**
```python
green_image = torch.tensor([
    [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],  # R (균일)
    [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]],  # G (균일)
    [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]   # B
])
```

---

## 4. 학습 과정 (Training Loop)

### 4.1 Forward Pass (순전파)
```python
z = torch.sum(image * w) + b
prediction = torch.sigmoid(z)  # 0~1 사이의 확률로 변환
```

**시그모이드 함수**: 어떤 값이든 0과 1 사이로 압축 → 확률 해석 가능

### 4.2 Loss 계산 (오차 측정)
```python
loss = (target - prediction) ** 2  # MSE (Mean Squared Error)
```
- 정답이 1인데 0.1로 예측 → loss = 0.81 (큰 오차)
- 정답이 1인데 0.9로 예측 → loss = 0.01 (작은 오차)

### 4.3 Backward Pass (역전파)
```python
loss.backward()  # 각 가중치가 오차에 얼마나 기여했는지 계산
print(w.grad)    # 가중치별 기울기 (책임 소재)
```

### 4.4 가중치 업데이트 (경사하강법)
```python
learning_rate = 0.5

with torch.no_grad():  # 업데이트 중에는 미분 기록 중지
    w -= w.grad * learning_rate  # 기울기 반대 방향으로 이동
    b -= b.grad * learning_rate
    w.grad.zero_()  # 다음 반복을 위해 기울기 초기화
    b.grad.zero_()
```

---

## 5. 완전한 학습 루프

```python
for i in range(5000):
    # Forward
    pred_brown = torch.sigmoid(torch.sum(brown_image * w) + b)
    pred_green = torch.sigmoid(torch.sum(green_image * w) + b)
    
    # Loss
    loss_brown = (1.0 - pred_brown) ** 2  # 갈색은 1이 정답
    loss_green = (0.0 - pred_green) ** 2  # 초록은 0이 정답
    total_loss = loss_brown + loss_green
    
    # Backward
    total_loss.backward()
    
    # Update
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        w.grad.zero_()
        b.grad.zero_()
```

**결과**: 학습 후 R채널 중앙 가중치가 8.16으로 매우 커짐 → 갈색 특징(R 높음) 감지

---

## 6. PyTorch 추상화 (nn.Module)

수동 구현 대신 PyTorch 제공 도구 사용:

```python
import torch.nn as nn
import torch.optim as optim

# 데이터 준비 (3x3x3 → 27개로 펼침)
X = torch.stack([brown_image.flatten(), green_image.flatten()])
Y = torch.tensor([[1.0], [0.0]])

# 모델 정의
model = nn.Sequential(
    nn.Linear(27, 1),  # 27개 입력 → 1개 출력 (w, b 자동 생성)
    nn.Sigmoid()
)

# 학습 도구
criterion = nn.MSELoss()                    # 손실 함수
optimizer = optim.SGD(model.parameters(), lr=0.5)  # 최적화 도구

# 학습 루프
for i in range(5000):
    prediction = model(X)           # Forward
    loss = criterion(prediction, Y) # Loss
    
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()        # 역전파
    optimizer.step()       # 가중치 업데이트 (w -= w.grad * lr)
```

---

## 핵심 정리

| 단계 | 역할 | 코드 |
|------|------|------|
| **Forward** | 입력 → 예측 | `pred = model(x)` |
| **Loss** | 예측 vs 정답 차이 | `loss = criterion(pred, y)` |
| **Backward** | 기울기 계산 | `loss.backward()` |
| **Update** | 가중치 수정 | `optimizer.step()` |

> 🔑 **핵심 인사이트**: 신경망은 결국 **"오차를 줄이는 방향으로 가중치를 조금씩 조정"**하는 과정을 반복하여 패턴을 학습한다.
