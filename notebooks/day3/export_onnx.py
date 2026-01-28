"""
PyTorch 모델 → ONNX 변환 스크립트

.pth 파일을 브라우저에서 실행 가능한 .onnx 파일로 변환합니다.
변환된 파일은 React/Next.js 프로젝트의 public/models/ 폴더에 복사하세요.

사용법:
    python export_onnx.py
    python export_onnx.py --input ../day2/circle_model_finetuned_best.pth --output circle_model.onnx
"""

import argparse
import os
import sys

import torch
import torch.onnx

# day2 모듈 import를 위해 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "day2"))
from circle import CircleRegressor


def export_to_onnx(pth_path, onnx_path, img_size=256):
    """
    PyTorch 모델을 ONNX 형식으로 변환
    
    Args:
        pth_path: 학습된 .pth 파일 경로
        onnx_path: 저장할 .onnx 파일 경로
        img_size: 입력 이미지 크기 (기본 256)
    """
    device = torch.device("cpu")
    
    # 1. 모델 객체 생성 (6채널: RGB + Edge + Coord)
    model = CircleRegressor(in_channels=6)
    
    # 2. 학습된 가중치 로드
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model.eval()  # 추론 모드 (Dropout, BatchNorm 고정)
    
    # 3. 가짜 입력 데이터 생성 (Batch, Channel, H, W)
    dummy_input = torch.randn(1, 6, img_size, img_size)
    
    # 4. ONNX 변환
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,           # 가중치 포함
        opset_version=12,             # ONNX Runtime Web 호환 버전
        do_constant_folding=True,     # 상수 폴딩 최적화
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    # 5. 파일 크기 확인
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"변환 완료: {onnx_path}")
    print(f"파일 크기: {file_size:.2f} MB")
    print(f"입력 형식: (batch, 6, {img_size}, {img_size})")
    print(f"출력 형식: (batch, 3) → [x_norm, y_norm, r_norm]")


def verify_onnx(onnx_path):
    """ONNX 모델 검증"""
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"ONNX 모델 검증 통과")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch → ONNX 변환")
    parser.add_argument(
        "--input", 
        default="../day2/circle_model_finetuned_best.pth",
        help="입력 .pth 파일 경로"
    )
    parser.add_argument(
        "--output", 
        default="circle_model.onnx",
        help="출력 .onnx 파일 경로"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="변환 후 ONNX 모델 검증"
    )
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    export_to_onnx(args.input, args.output)
    
    if args.verify:
        verify_onnx(args.output)
