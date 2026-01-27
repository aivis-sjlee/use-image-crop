import torch
import torch.nn as nn
from data_factory import CircleDataset, IMG_SIZE
from torch.utils.data import DataLoader
import torch.optim as optim

# 데이터 3000개 (정확도 향상)
dataset = CircleDataset(num_samples=3000)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# [1] 모델 설계: 경량 CNN + Global Average Pooling
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
    
    # Global Average Pooling (3층으로 충분)
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    
    # Regression Head
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) # 학습률은 0.001이 국룰!

epochs = 80
print("--- 훈련 시작! ---")
print(f"이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print(f"데이터셋: {len(dataset)}개, 배치: 64")
print(f"에포크: {epochs}회")
print()

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in loader: # 공장에서 만든 이미지와 정답(x,y,r) 꺼내기
        # 1. 변화도 초기화 (이전 학습의 영향을 지움)
        optimizer.zero_grad()
        # 2. 순전파: 모델의 예측값 계산
        outputs = model(inputs)
        # 3. 손실 계산: 예측값과 실제 정답의 거리 측정
        loss = criterion(outputs, labels)
        # 4. 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.6f}")

print("--- 훈련 완료! ---")

# 모델 저장
torch.save(model.state_dict(), 'circle_model.pth')
print("\n모델 저장 완료: circle_model.pth")

# 테스트용 데이터 하나 추출
test_img, test_label = dataset[0] # 첫 번째 데이터 꺼내기
model.eval() # 평가 모드 전환

with torch.no_grad():
    prediction = model(test_img.unsqueeze(0)) # 1장만 넣을 때는 차원을 하나 늘려줌

print(f"\n실제 정답 (x, y, r): {test_label.numpy()}")
print(f"모델 예측 (x, y, r): {prediction.squeeze().numpy()}")