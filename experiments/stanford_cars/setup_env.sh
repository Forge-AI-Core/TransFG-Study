#!/bin/bash
# TransFG Stanford Cars 가상환경 설정 스크립트
# 환경: macOS (CPU/MPS) | Ubuntu x86_64 (CUDA 12.1) | Ubuntu aarch64 (CUDA 13.0, NVIDIA GB10 등)
# 사용법: bash setup_env.sh

set -e

VENV_NAME="TransFG_cars_venv"
PYTHON="python3"

echo "=========================================="
echo " TransFG Stanford Cars 가상환경 설정"
echo "=========================================="

# [1/5] 가상환경 생성
echo "[1/5] 가상환경 생성: $VENV_NAME"
$PYTHON -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# [2/5] pip 업그레이드
echo "[2/5] pip 업그레이드"
pip install --upgrade pip setuptools wheel

# [3/5] PyTorch 설치 (OS/아키텍처 분기)
echo "[3/5] PyTorch 설치"
ARCH="$(uname -m)"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (Apple Silicon MPS 지원)
    echo "  → macOS 감지: CPU/MPS PyTorch 설치"
    pip install torch torchvision torchaudio

elif [[ "$ARCH" == "aarch64" ]]; then
    # Linux aarch64 (NVIDIA GB10 / DGX Spark 등)
    echo "  → Linux aarch64 감지: CUDA 13.0 PyTorch 설치"
    pip install "torch==2.10.0+cu130" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

else
    # Linux x86_64 (일반 CUDA GPU)
    echo "  → Linux x86_64 감지: CUDA 12.1 PyTorch 설치"
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
fi

# [4/5] 프로젝트 의존성 설치
echo "[4/5] 의존성 설치 (requirements.txt)"
pip install -r requirements.txt

# [5/5] Jupyter 커널 등록
echo "[5/5] Jupyter 커널 등록"
pip install ipykernel
python -m ipykernel install --user --name=$VENV_NAME --display-name "TransFG Cars"

echo ""
echo "=========================================="
echo " ✅ 설치 완료!"
echo "=========================================="
echo "  가상환경 활성화 : source $VENV_NAME/bin/activate"
echo "  Jupyter 커널    : 'TransFG Cars' 선택"
echo "  PyCharm         : Interpreter를 ./$VENV_NAME/bin/python 으로 설정"
echo ""
echo "📂 다음 단계 (README.md 참고):"
echo "  1) 데이터 준비 (HuggingFace tanganke/stanford_cars):"
echo "     pip install -U \"huggingface_hub[cli]\""
echo "     huggingface-cli download tanganke/stanford_cars \\"
echo "       --repo-type dataset --local-dir ./Stanford_Cars"
echo ""
echo "  2) 사전학습 가중치 준비 (Phase 2 - 둘 중 하나):"
echo "     # ImageNet baseline (200 step 비교용)"
echo "     mkdir -p pretrained"
echo "     wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P pretrained/"
echo "     # Aircraft 학습 결과 (Phase 2 메인 실험)"
echo "     # → feature/SCRUM-24 브랜치 TransFG_Aircraft_New.ipynb로 학습 후 결과 체크포인트를"
echo "     #   pretrained/aircraft_pretrained.bin 으로 복사"
echo ""
echo "  3) 노트북 실행:"
echo "     jupyter notebook TransFG_Stanford_Cars_New.ipynb"
echo "=========================================="
