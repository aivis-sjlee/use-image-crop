#!/bin/bash

# 스크립트 위치로 이동
cd "$(dirname "$0")"

# 버전 번호 찾기 (가장 높은 버전 + 1)
find_next_version() {
    local prefix=$1
    local max_version=0
    
    for file in ${prefix}_v*.pth; do
        if [[ -f "$file" ]]; then
            version=$(echo "$file" | sed "s/${prefix}_v\([0-9]*\)\.pth/\1/")
            if [[ "$version" -gt "$max_version" ]]; then
                max_version=$version
            fi
        fi
    done
    
    echo $((max_version + 1))
}

# 새 버전 번호 생성
VERSION=$(find_next_version "circle_model")
export MODEL_VERSION=$VERSION
export BASE_MODEL="circle_model_v${VERSION}.pth"
export FINETUNED_MODEL="circle_model_finetuned_v${VERSION}.pth"

echo "========================================"
echo "버전: v${VERSION}"
echo "기본 모델: ${BASE_MODEL}"
echo "Fine-tuned 모델: ${FINETUNED_MODEL}"
echo "========================================"

START_TIME=$(date +%s)

echo ""
echo "========================================"
echo "1단계: 기본 모델 학습 (circle.py)"
echo "========================================"
STEP1_START=$(date +%s)
python circle.py
STEP1_END=$(date +%s)
echo "1단계 소요: $((STEP1_END - STEP1_START))초"

echo ""
echo "========================================"
echo "2단계: Fine-tuning (finetune.py)"
echo "========================================"
STEP2_START=$(date +%s)
python finetune.py
STEP2_END=$(date +%s)
echo "2단계 소요: $((STEP2_END - STEP2_START))초"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "완료! 모델 파일 (v${VERSION}):"
echo "  - ${BASE_MODEL}"
echo "  - ${FINETUNED_MODEL}"
echo "========================================"
echo "총 소요 시간: ${TOTAL_TIME}초 ($((TOTAL_TIME / 60))분 $((TOTAL_TIME % 60))초)"
