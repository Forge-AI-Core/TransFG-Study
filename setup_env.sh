#!/bin/bash
# TransFG 가상환경 설정 스크립트 (macOS / Ubuntu)
# 사용법: bash setup_env.sh

set -e

VENV_NAME="transfg_env"
PYTHON="python3"

echo "=========================================="
echo " TransFG 가상환경 설정"
echo "=========================================="

# 가상환경 생성
echo "[1/5] 가상환경 생성: $VENV_NAME"
$PYTHON -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# pip 업그레이드
echo "[2/5] pip 업그레이드"
pip install --upgrade pip setuptools wheel

# PyTorch 설치 (OS별 분기)
echo "[3/5] PyTorch 설치"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (MPS 지원 포함)
    echo "  → macOS 감지: CPU/MPS 지원 PyTorch 설치"
    pip install torch torchvision torchaudio
else
    # Ubuntu / Linux (CUDA)
    echo "  → Linux 감지: CUDA 지원 PyTorch 설치"
    # CUDA 11.8
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # CUDA 12.1 (최신 권장)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# 의존성 설치
echo "[4/5] 의존성 설치 (requirements_updated.txt)"
pip install -r requirements_updated.txt

# Jupyter 커널 등록 (PyCharm에서 해당 venv를 커널로 선택 가능)
echo "[5/5] Jupyter 커널 등록"
pip install ipykernel
python -m ipykernel install --user --name=$VENV_NAME --display-name "TransFG (Python)"

echo ""
echo "=========================================="
echo " 설치 완료!"
echo "  가상환경 활성화: source $VENV_NAME/bin/activate"
echo "  PyCharm: Interpreter를 ./$VENV_NAME/bin/python 으로 설정"
echo "=========================================="
