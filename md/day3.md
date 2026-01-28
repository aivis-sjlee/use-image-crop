# Day 3: 모델 배포 (PyTorch → ONNX → React)

## 목표

Python에서 학습한 `.pth` 모델을 브라우저에서 실행 가능한 `.onnx`로 변환하여 React/Next.js에서 사용한다.

## 왜 ONNX인가?

| 형식 | 실행 환경 | 용도 |
|------|----------|------|
| `.pth` | Python (PyTorch) | 학습/연구 |
| `.onnx` | 브라우저, 모바일, 서버 | 배포/서비스 |

ONNX(Open Neural Network Exchange)는 딥러닝 모델의 **표준 교환 형식**이다. PyTorch, TensorFlow 등 어떤 프레임워크에서 만들었든 ONNX로 변환하면 어디서든 실행 가능하다.

## 변환 과정

### 1. 변환 스크립트 (`export_onnx.py`)

```python
import torch
from circle import CircleRegressor

def export_to_onnx(pth_path, onnx_path):
    # 1. 모델 구조 생성 (6채널 입력)
    model = CircleRegressor(in_channels=6)
    
    # 2. 학습된 가중치 로드
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()  # 추론 모드 (Dropout, BatchNorm 고정)
    
    # 3. 더미 입력 생성 (모델 입력 규격 정의용)
    dummy_input = torch.randn(1, 6, 256, 256)
    
    # 4. ONNX 변환
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,        # 가중치 포함
        opset_version=12,          # ONNX Runtime Web 호환
        do_constant_folding=True,  # 상수 최적화
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={             # 배치 크기 유동적
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
```

### 2. 실행

```bash
cd notebooks/day3
pip install onnx  # 최초 1회
python export_onnx.py
```

출력: `circle_model.onnx` (약 1~2MB)

## React/Next.js 배포

### 1. 파일 복사

```bash
cp circle_model.onnx /path/to/nextjs-project/public/models/
```

### 2. 패키지 설치

```bash
npm install onnxruntime-web
```

### 3. 전처리 함수 (6채널 생성)

Python의 `preprocess_image`를 JavaScript로 똑같이 구현해야 한다.

```typescript
// utils/preprocess.ts

export function preprocessImage(imageData: ImageData, targetSize = 256): Float32Array {
  const tensor = new Float32Array(6 * targetSize * targetSize);
  
  // 1. 리사이즈 + 패딩
  // 2. RGB 채널 (0~1 정규화)
  // 3. Edge 채널 (Sobel 필터)
  // 4. Coord 채널 (x, y 좌표)
  
  return tensor;  // [1, 6, 256, 256] 형태
}
```

**핵심**: Python에서 6채널(RGB + Edge + Coord)로 학습했으므로, JS에서도 **동일한 전처리**를 해야 한다.

### 4. 모델 실행

```typescript
// utils/inference.ts
import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;

export async function loadModel() {
  if (!session) {
    session = await ort.InferenceSession.create('/models/circle_model.onnx');
  }
  return session;
}

export async function predict(inputTensor: Float32Array) {
  const session = await loadModel();
  const tensor = new ort.Tensor('float32', inputTensor, [1, 6, 256, 256]);
  const results = await session.run({ input: tensor });
  
  const output = results.output.data as Float32Array;
  return {
    x: output[0],  // 0~1 정규화된 X 좌표
    y: output[1],  // 0~1 정규화된 Y 좌표
    r: output[2],  // 0~1 정규화된 반지름
  };
}
```

### 5. 좌표 역변환

모델 출력은 256x256 정사각형 기준 정규화 좌표이므로, 원본 이미지 좌표로 변환해야 한다.

```typescript
export function denormalize(
  pred: { x: number; y: number; r: number },
  meta: { origW: number; origH: number; scale: number; padX: number; padY: number }
) {
  const xPx = pred.x * 256;
  const yPx = pred.y * 256;
  const rPx = pred.r * 256;
  
  return {
    x: Math.round((xPx - meta.padX) / meta.scale),
    y: Math.round((yPx - meta.padY) / meta.scale),
    r: Math.round(rPx / meta.scale),
  };
}
```

## 파일 구조

```
notebooks/day3/
├── export_onnx.py       # 변환 스크립트
├── circle_model.onnx    # 변환된 모델
└── README.md            # React 통합 가이드

nextjs-project/
├── public/models/
│   └── circle_model.onnx
├── utils/
│   ├── preprocess.ts    # 6채널 전처리
│   └── inference.ts     # ONNX 실행
└── components/
    └── CircleDetector.tsx
```

## 핵심 포인트

1. **eval() 필수**: 변환 전 `model.eval()` 호출하여 Dropout, BatchNorm을 추론 모드로 고정
2. **dummy_input 규격**: 실제 입력과 동일한 shape 사용 `(1, 6, 256, 256)`
3. **전처리 일치**: Python과 JavaScript의 전처리 로직이 **완전히 동일**해야 함
4. **opset_version=12**: onnxruntime-web과 호환성 좋은 버전

## 다음 단계

- React 컴포넌트에서 카메라/파일 업로드 연동
- Canvas에 검출된 원 오버레이 표시
- 실시간 검출 성능 최적화
