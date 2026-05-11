# 환경 설정 가이드

## 요구 사항

| 항목 | 버전 |
|------|------|
| Python | 3.9.x 권장 |
| PyTorch | 2.0+ |
| OS | macOS (MPS) / Ubuntu (CUDA) |

---

## macOS (Apple Silicon M1/M2/M3/M4)

```bash
# 1. 저장소 클론
git clone <repo-url>
cd TransFG

# 2. 가상환경 생성
python3.9 -m venv TransFG_venv
source TransFG_venv/bin/activate

# 3. PyTorch 설치 (MPS 지원)
pip install torch torchvision

# 4. 나머지 패키지 설치
pip install -r requirements_updated.txt

# 5. Jupyter 커널 등록
pip install ipykernel
python -m ipykernel install --user --name TransFG_venv --display-name "TransFG (Python 3.9)"
```

---

## Ubuntu / Linux (CUDA GPU)

```bash
# 1. 저장소 클론
git clone <repo-url>
cd TransFG

# 2. 가상환경 생성
python3.9 -m venv TransFG_venv
source TransFG_venv/bin/activate

# 3. PyTorch 설치 (CUDA 12.x 기준)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 나머지 패키지 설치
pip install -r requirements_updated.txt

# 5. Jupyter 커널 등록
pip install ipykernel
python -m ipykernel install --user --name TransFG_venv --display-name "TransFG (Python 3.9)"
```

---

## 데이터셋 준비

### CUB-200-2011 (약 1.2GB)

```bash
# 방법 1: 공식 사이트에서 수동 다운로드
# https://www.vision.caltech.edu/datasets/cub-200-2011/
# CUB_200_2011.tgz 다운로드 후:

mkdir -p CUB_200_2011
tar -xzf CUB_200_2011.tgz
# 중첩 폴더 없이 CUB_200_2011/ 바로 아래 images/, classes.txt 등이 있어야 함

# 최종 구조 확인:
# TransFG/CUB_200_2011/
# ├── images/
# ├── classes.txt
# ├── image_class_labels.txt
# ├── images.txt
# ├── train_test_split.txt
# └── bounding_boxes.txt
```

### Pretrained ViT-B_16 (약 394MB)

```bash
mkdir -p pretrained
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P pretrained/
# 또는 curl 사용:
# curl -o pretrained/ViT-B_16.npz https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

---

## 동작 확인

```bash
source TransFG_venv/bin/activate
python -c "
import torch
print('PyTorch:', torch.__version__)
if torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
elif torch.cuda.is_available():
    print('Device: CUDA', torch.cuda.get_device_name(0))
else:
    print('Device: CPU')
"
```

---

## 노트북 실행

```bash
source TransFG_venv/bin/activate
jupyter notebook TransFG_CUB200.ipynb
# 커널: "TransFG (Python 3.9)" 선택
```

---

## 문제 해결

### `ModuleNotFoundError: scipy`
```bash
pip install scipy
```

### `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'`
```bash
# Pillow 버전 문제 - requirements_updated.txt에 이미 반영됨
pip install "Pillow<10.0.0"
```

### MPS에서 학습이 느린 경우
- MPS는 CUDA 대비 약 10~20배 느림
- Cell 12의 빠른 테스트 (200 steps)로 동작 확인 후
- 실제 전체 학습은 CUDA 환경(Ubuntu) 권장
