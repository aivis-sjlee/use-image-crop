# Day 3: 모델 배포 (PyTorch → ONNX → React)

## 1. ONNX 변환

```bash
cd notebooks/day3
python export_onnx.py
```

출력: `circle_model.onnx` (약 1~5MB)

## 2. React/Next.js 프로젝트에 배포

```bash
# ONNX 파일을 Next.js public 폴더로 복사
cp circle_model.onnx /path/to/your-nextjs-project/public/models/
```

## 3. React에서 사용하기

### 패키지 설치

```bash
npm install onnxruntime-web
```

### 전처리 함수 (6채널 생성)

```typescript
// utils/preprocess.ts

/**
 * 이미지를 모델 입력 형식 (1, 6, 256, 256)으로 변환
 * 채널: RGB(3) + Edge(1) + Coord(2)
 */
export function preprocessImage(
  imageData: ImageData,
  targetSize: number = 256
): Float32Array {
  const { width, height, data } = imageData;
  
  // 1. 리사이즈 + 패딩 (정사각형으로)
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d')!;
  
  const scale = targetSize / Math.max(width, height);
  const newW = Math.round(width * scale);
  const newH = Math.round(height * scale);
  const padX = Math.floor((targetSize - newW) / 2);
  const padY = Math.floor((targetSize - newH) / 2);
  
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(/* source image */, padX, padY, newW, newH);
  
  const resizedData = ctx.getImageData(0, 0, targetSize, targetSize).data;
  
  // 2. 6채널 텐서 생성
  const tensor = new Float32Array(6 * targetSize * targetSize);
  
  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {
      const i = (y * targetSize + x) * 4;
      const j = y * targetSize + x;
      
      // RGB (0~1 정규화)
      tensor[0 * targetSize * targetSize + j] = resizedData[i] / 255;     // R
      tensor[1 * targetSize * targetSize + j] = resizedData[i + 1] / 255; // G
      tensor[2 * targetSize * targetSize + j] = resizedData[i + 2] / 255; // B
      
      // Coord (위치 정보)
      tensor[4 * targetSize * targetSize + j] = x / targetSize; // X coord
      tensor[5 * targetSize * targetSize + j] = y / targetSize; // Y coord
    }
  }
  
  // 3. Edge 채널 (Sobel 근사)
  computeEdgeChannel(resizedData, tensor, targetSize);
  
  return tensor;
}

function computeEdgeChannel(
  data: Uint8ClampedArray, 
  tensor: Float32Array, 
  size: number
) {
  // Grayscale 변환
  const gray = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) {
    const idx = i * 4;
    gray[i] = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114) / 255;
  }
  
  // Sobel 필터 적용
  let maxMag = 0;
  const edge = new Float32Array(size * size);
  
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const idx = y * size + x;
      
      // Sobel X
      const gx = 
        -gray[(y - 1) * size + (x - 1)] + gray[(y - 1) * size + (x + 1)] +
        -2 * gray[y * size + (x - 1)] + 2 * gray[y * size + (x + 1)] +
        -gray[(y + 1) * size + (x - 1)] + gray[(y + 1) * size + (x + 1)];
      
      // Sobel Y
      const gy = 
        -gray[(y - 1) * size + (x - 1)] - 2 * gray[(y - 1) * size + x] - gray[(y - 1) * size + (x + 1)] +
        gray[(y + 1) * size + (x - 1)] + 2 * gray[(y + 1) * size + x] + gray[(y + 1) * size + (x + 1)];
      
      edge[idx] = Math.sqrt(gx * gx + gy * gy);
      maxMag = Math.max(maxMag, edge[idx]);
    }
  }
  
  // 정규화 후 텐서에 저장
  const edgeOffset = 3 * size * size;
  for (let i = 0; i < size * size; i++) {
    tensor[edgeOffset + i] = edge[i] / (maxMag + 1e-6);
  }
}
```

### ONNX 모델 실행

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

### 좌표 역변환 (원본 이미지 기준)

```typescript
export function denormalize(
  pred: { x: number; y: number; r: number },
  meta: { origW: number; origH: number; scale: number; padX: number; padY: number },
  targetSize: number = 256
) {
  const xPx = pred.x * targetSize;
  const yPx = pred.y * targetSize;
  const rPx = pred.r * targetSize;
  
  return {
    x: Math.round((xPx - meta.padX) / meta.scale),
    y: Math.round((yPx - meta.padY) / meta.scale),
    r: Math.round(rPx / meta.scale),
  };
}
```

## 파일 구조

```
your-nextjs-project/
├── public/
│   └── models/
│       └── circle_model.onnx  ← 여기에 복사
├── utils/
│   ├── preprocess.ts
│   └── inference.ts
└── components/
    └── CircleDetector.tsx
```
