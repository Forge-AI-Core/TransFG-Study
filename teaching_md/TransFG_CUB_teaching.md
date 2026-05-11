# TransFG로 새 200종 분류하기 — 대학 1학년을 위한 친절한 가이드

> 노트북: `TransFG_CUB.ipynb`
> 논문: TransFG (AAAI 2022) — CUB-200-2011 dataset

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [핵심 개념 (비유로 이해하기)](#2-핵심-개념-비유로-이해하기)
3. [전체 구조 한눈에 보기 (ASCII)](#3-전체-구조-한눈에-보기-ascii)
4. [셀별 코드 설명](#4-셀별-코드-설명)
   - [Cell 0 — 숨겨진 첫 셀](#cell-0--숨겨진-첫-셀-python-인터프리터-확인)
   - [Cell 1 — 환경 확인](#cell-1--환경-확인)
   - [Cell 2 — 패키지 설치](#cell-2--패키지-설치)
   - [Cell 3 — 데이터셋 다운로드](#cell-3--데이터셋-다운로드)
   - [Cell 4 — Pretrained 가중치 다운로드](#cell-4--pretrained-가중치-다운로드)
   - [Cell 5 — 모듈 임포트 및 경로 설정](#cell-5--모듈-임포트-및-경로-설정)
   - [Cell 6 — 데이터셋 탐색](#cell-6--데이터셋-탐색)
   - [Cell 7 — 샘플 이미지 시각화](#cell-7--샘플-이미지-시각화)
   - [Cell 8 — 하이퍼파라미터 설정](#cell-8--하이퍼파라미터-설정)
   - [Cell 9 — 데이터로더 생성](#cell-9--데이터로더-생성)
   - [Cell 10 — 모델 생성](#cell-10--모델-생성)
   - [Cell 11 — 모델 구조 탐색](#cell-11--모델-구조-탐색)
   - [Cell 12 — 빠른 학습 테스트](#cell-12--빠른-학습-테스트-quick-training-200-steps)
   - [Cell 13 — 학습 history 시각화](#cell-13--학습-history-시각화)
   - [Cell 14 — 전체 학습](#cell-14--전체-학습-주석-처리됨)
   - [Cell 15 — 체크포인트 로드](#cell-15--체크포인트-로드)
   - [Cell 16 — 테스트셋 평가](#cell-16--테스트셋-평가)
   - [Cell 17 — 단일 이미지 추론](#cell-17--단일-이미지-추론)
   - [Cell 18 — 배치 예측 시각화](#cell-18--배치-예측-시각화)
   - [Cell 19 — Self-Attention Map 시각화](#cell-19--self-attention-map-시각화)
   - [Cell 20 — 12개 Head Attention 비교](#cell-20--12개-head-attention-비교)
5. [전체 셀 요약 표](#5-전체-셀-요약-표)
6. [공부 팁](#6-공부-팁)

---

## 1. 한 줄 요약

> 새(Bird) 사진을 보고 200종 중 어떤 종(species)인지 맞추는 인공지능을, **ViT(Vision Transformer)** 모델에 **Part Selection Module**을 붙인 **TransFG** 논문 방식으로 학습시켜보는 실습이다.

---

## 2. 핵심 개념 (비유로 이해하기)

### 2-1. ViT (Vision Transformer)

```
[이미지]               [패치 분할]              [언어 모델처럼 처리]
┌─────────┐           ┌──┬──┬──┬──┐
│  새 사진 │   →      │P1│P2│P3│P4│   →     "P1 P2 P3 P4 ..."
│         │           ├──┼──┼──┼──┤             ↓
│         │           │P5│P6│P7│P8│         Transformer Encoder
└─────────┘           └──┴──┴──┴──┘             ↓
                       16×16 패치들              [클래스 예측]
```

- **비유**: 이미지를 직소 퍼즐 조각처럼 16×16 픽셀 작은 조각(patch)으로 자른다.
- 각 조각을 NLP에서 **단어(token)** 처럼 처리한다.
- 즉, "이미지 = 패치들의 문장"으로 보고 Transformer로 분석.

### 2-2. Overlap Patch Embedding

```
일반 ViT (stride=16):              TransFG Overlap (stride=12):
┌────┬────┬────┐                  ┌────┐
│ P1 │ P2 │ P3 │                  │ P1 │
└────┴────┴────┘                  └─┬──┘
겹침 없음 → 14×14=196               └─┌────┐
                                     │ P2 │ ← 4픽셀씩 겹침
                                     └─┬──┘
                                       └─┌────┐
                                         │ P3 │
                                         └────┘
                                  더 촘촘 → 37×37=1,369
```

- **비유**: 사진을 잘라낼 때 일반 ViT는 가위로 깨끗하게 자르고, TransFG는 일부러 **조각끼리 겹치게** 자른다.
- **왜?** 새 부리 같은 미세한 부분이 패치 경계에 끊겨버리는 걸 방지하기 위해서이다.
- 결과: 224×224 → 14×14 patches가, 448×448 + stride=12 → 37×37=**1,369 patches** 가 된다.

### 2-3. Part Selection Module (PSM)

```
Layer 0  Layer 1  Layer 2  ...  Layer 10        가장 중요한 patch
   ↓        ↓        ↓             ↓                  ↓
 [attn] × [attn] × [attn] × ... × [attn]   →   [Top-K patch 선택]
   누적 곱 (matrix multiply)               → 마지막 Block에 입력
```

- **비유**: 11명의 심사위원이 각자 점수를 매긴다 → 점수를 **곱하기**로 합쳐서 진짜 중요한 부분만 추린다.
- 새의 종 구분에는 **부리, 눈, 날개 무늬** 같은 미세 영역이 결정적이므로, 전체 1,369 patches 중 가장 중요한 일부만 골라서 정밀하게 본다.

### 2-4. Fine-Grained Recognition

```
일반 분류:                          Fine-Grained 분류 (TransFG):
┌────────┐                        ┌─────────────────────┐
│ 새 vs 개│                        │ 200종 새 중에서      │
│ vs 고양이│                        │  - Black Albatross  │
└────────┘                        │  - Sooty Albatross  │
                                  │  ... (거의 똑같음!) │
                                  └─────────────────────┘
```

- **비유**: 일반 분류는 "강아지인가 고양이인가"이고, Fine-Grained는 "푸들인가 말티즈인가"이다.
- 같은 새(Bird)인데 종이 다른 것을 구분 — 인간 눈으로도 어렵다.

### 2-5. Pretrained Weight (전이학습)

- **비유**: 영어를 이미 잘하는 사람이 한국어를 배우면 처음부터 배우는 사람보다 빠르다.
- ImageNet-21k(1,400만 장)로 미리 훈련된 ViT-B/16 가중치를 가져와서, CUB-200 새 데이터로 추가 학습.

### 2-6. GPU vs CPU

| 특징     | CPU                    | GPU                          |
|----------|------------------------|------------------------------|
| 코어 수   | 수십 개                | 수천 개                       |
| 잘하는 일 | 복잡한 순차 처리        | 단순한 병렬 행렬 연산          |
| 딥러닝   | 느림 (수일~수주)        | 빠름 (수시간) ← 이 실습에서 사용 |

### 2-7. FP16 (Mixed Precision)

| 정밀도 | 비트 수 | 메모리 | 속도 | 정확도 |
|--------|---------|--------|------|--------|
| FP32   | 32-bit  | 100%   | 1x   | 기준    |
| FP16   | 16-bit  | 50%    | ~2x  | 거의 동일 |

- **비유**: 자(ruler)를 mm 단위 → cm 단위로 바꾸는 것과 비슷. 정밀도는 약간 떨어지지만 충분히 유용하고 빠르다.

---

## 3. 전체 구조 한눈에 보기 (ASCII)

```
┌────────────────────────────────────────────────────────────────────┐
│                    TransFG_CUB.ipynb 전체 흐름                      │
└────────────────────────────────────────────────────────────────────┘

[준비 단계]                       [학습 준비]                  [학습/평가]
─────────────                  ─────────────              ─────────────
Cell 0  Python 확인              Cell 5  import           Cell 12 Quick 학습
Cell 1  GPU/CUDA 확인            Cell 6  데이터 통계      Cell 13 history 시각화
Cell 2  패키지 설치              Cell 7  샘플 이미지       Cell 14 (전체 학습)
Cell 3  CUB 데이터 다운로드       Cell 8  하이퍼파라미터    Cell 15 체크포인트 로드
Cell 4  ViT 가중치 다운로드       Cell 9  DataLoader       Cell 16 테스트 평가
                                Cell 10 모델 생성         Cell 17 단일 이미지 추론
                                Cell 11 모델 구조 분석    Cell 18 배치 시각화
                                                         Cell 19 Attention map
                                                         Cell 20 12-head 비교

  [데이터]                        [모델 (TransFG)]               [결과]
   ─────                           ───────────                    ────
                                ┌──────────────┐
   CUB-200      ──────►        │ Patch Embed  │           ──►  200 클래스
   11,788장                    │ (overlap)    │               확률 분포
   200종 새                    └──────┬───────┘
                                       │
                                ┌──────▼───────┐
                                │ Encoder      │
                                │ (11 blocks)  │
                                └──────┬───────┘
                                       │ attention
                                ┌──────▼───────┐
                                │ Part Select  │
                                │ (PSM)        │
                                └──────┬───────┘
                                       │
                                ┌──────▼───────┐
                                │ Final Block  │
                                │ + Head       │
                                └──────────────┘
```

---

## 4. 셀별 코드 설명

### Cell 0 — 숨겨진 첫 셀 (Python 인터프리터 확인)

```
[목적]
 ─────
 어떤 Python 환경에서 노트북이 돌아가는지 확인
```

```python
import sys
print(sys.executable)
print(sys.version)
```

**실제 출력:**
```
/home/changilkim/Documents/aiffel_class/AiffelThon01/TransFG_venv/bin/python
3.12.3 (main, Jan 22 2026, 20:57:42) [GCC 13.3.0]
```

**코드 설명:**
- `import sys`: 파이썬 시스템 정보를 다루는 모듈
- `sys.executable`: 현재 실행 중인 파이썬 인터프리터 경로 → 가상환경(venv) 사용 여부 확인
- `sys.version`: 파이썬 버전 정보

**왜 이렇게 하는가?**
- 노트북 커널이 잘못 잡히면 패키지가 안 보일 수 있다 → 가장 먼저 환경 확인.
- `TransFG_venv`라는 별도 가상환경에서 돌아가고 있음을 확인할 수 있다.

---

### Cell 1 — 환경 확인

```
[목적]                       [확인 항목]
 ─────                        ──────────
 GPU 사용 가능 여부 검증    →  Python / PyTorch / CUDA / GPU 이름 / 메모리
```

```python
import sys
import torch

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"Memory : {total_mem / 1e9:.1f} GB")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
```

**실제 출력:**
```
Python : 3.12.3
PyTorch: 2.10.0+cu130
CUDA   : True
GPU    : NVIDIA GB10
Memory : 130.7 GB
Compute Capability: 12.1
```

**코드 설명:**
- `import torch`: PyTorch 딥러닝 라이브러리
- `torch.cuda.is_available()`: GPU 사용 가능하면 `True`
- `torch.cuda.get_device_name(0)`: 0번 GPU의 이름
- `torch.cuda.get_device_properties(0).total_memory`: 전체 GPU 메모리 (바이트)
- `total_mem / 1e9`: 바이트를 GB로 환산 (10^9 = 1 GB)
- `torch.cuda.get_device_capability(0)`: GPU의 compute capability (블랙웰=12.x)

**왜 이렇게 하는가?**
- GPU가 없으면 학습이 수십~수백 배 느려진다 → **가장 먼저 확인**해야 한다.
- 메모리가 부족하면 batch size를 줄여야 하므로 미리 알아둔다.

---

### Cell 2 — 패키지 설치

```
[설치 흐름]
 ─────────
 venv 활성화 → pip install → 노트북 실행
```

```python
# TransFG_venv 커널 선택 시 이미 설치되어 있음
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install",
#     "torch==2.10.0+cu130", ...], check=True)
print("TransFG_venv 패키지 설치 완료 상태입니다.")
```

**실제 출력:**
```
TransFG_venv 패키지 설치 완료 상태입니다.
```

**코드 설명:**
- 주석 처리된 코드는 `pip install`을 노트북에서 실행하는 방법
- 가상환경에 이미 패키지가 다 깔려 있어서 메시지만 출력

**왜 이렇게 하는가?**
- 매번 노트북 실행할 때마다 패키지를 설치하면 시간 낭비 → 가상환경 한 번 만들어 놓고 재사용.
- `subprocess.run([sys.executable, "-m", "pip", ...])`은 정확히 **현재 노트북의 파이썬에** 설치하는 안전한 방식.

---

### Cell 3 — 데이터셋 다운로드

```
[다운로드 흐름]
 ─────────────
 URL  →  .tgz 파일  →  압축 해제  →  data/CUB_200_2011/
        (1.1 GB)              (이미지 11,788장)
```

```python
import os, tarfile, urllib.request
from pathlib import Path

DATA_ROOT = Path("data/CUB_200_2011")
URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
TGZ = Path("data/CUB_200_2011.tgz")

if (DATA_ROOT / "images.txt").exists():
    print(f"데이터셋이 이미 존재합니다: {DATA_ROOT.resolve()}")
else:
    os.makedirs("data", exist_ok=True)
    print(f"다운로드 중... (약 1.1 GB)\n{URL}")
    downloaded = [0]
    def show_progress(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(done / total_size * 100, 100)
            if block_num % 500 == 0:
                print(f"\r  {pct:.1f}%  ({done/1e6:.0f}/{total_size/1e6:.0f} MB)", end="", flush=True)
    urllib.request.urlretrieve(URL, TGZ, reporthook=show_progress)
    print("\n압축 해제 중...")
    with tarfile.open(TGZ, "r:gz") as tar:
        tar.extractall("data")
    TGZ.unlink()
    print(f"완료: {DATA_ROOT.resolve()}")

print("\n─ 데이터셋 파일 확인 ─")
for fname in ["images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"]:
    ok = (DATA_ROOT / fname).exists()
    print(f"  {'✓' if ok else '✗'} {fname}")
n_images = sum(1 for _ in (DATA_ROOT / "images").rglob("*.jpg")) if (DATA_ROOT / "images").exists() else 0
print(f"  총 이미지 수: {n_images:,}장")
```

**실제 출력:**
```
데이터셋이 이미 존재합니다: /home/changilkim/Documents/aiffel_class/AiffelThon01/data/CUB_200_2011
─ 데이터셋 파일 확인 ─
  ✓ images.txt
  ✓ image_class_labels.txt
  ✓ train_test_split.txt
  ✓ classes.txt
  총 이미지 수: 11,788장
```

**코드 설명 (핵심):**
- `Path("data/CUB_200_2011")`: `pathlib`로 경로를 객체처럼 다룸 (문자열 합치기보다 안전)
- `if (DATA_ROOT / "images.txt").exists()`: 이미 다운로드된 파일이 있으면 skip
- `urllib.request.urlretrieve(URL, TGZ, reporthook=show_progress)`: URL → 로컬 파일로 다운로드, 진행률 콜백 등록
- `tarfile.open(TGZ, "r:gz")`: gzip으로 압축된 tar 파일 열기 → `extractall("data")`로 압축 해제
- `TGZ.unlink()`: tgz 원본 파일 삭제 (디스크 공간 절약)
- `rglob("*.jpg")`: 하위 폴더까지 재귀적으로 jpg 파일 찾기

**왜 이렇게 하는가?**
- 다운로드는 한 번만 하면 되므로 `if exists()`로 중복 방지.
- `reporthook` 콜백으로 진행률 표시 → 사용자가 진행 상황을 알 수 있음.
- CUB-200-2011은 200종 새, 11,788장으로 구성된 fine-grained 분류의 표준 벤치마크 데이터셋.

---

### Cell 4 — Pretrained 가중치 다운로드

```
[가중치 출처]
 ───────────
 Google ViT (ImageNet-21k pretrained, 1,400만 장)
        ↓
 ViT-B_16.npz  (~413 MB)
        ↓
 우리 모델의 출발점
```

```python
import urllib.request
from pathlib import Path

PRETRAINED_DIR = Path("pretrained")
PRETRAINED_DIR.mkdir(exist_ok=True)
VIT_URL  = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
VIT_PATH = PRETRAINED_DIR / "ViT-B_16.npz"

if VIT_PATH.exists():
    size_mb = VIT_PATH.stat().st_size / 1e6
    print(f"가중치 파일 이미 존재: {VIT_PATH}  ({size_mb:.0f} MB)")
else:
    print(f"다운로드 중... (약 330 MB)\n{VIT_URL}")
    def show_progress(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0 and block_num % 1000 == 0:
            pct = min(done / total_size * 100, 100)
            print(f"\r  {pct:.1f}%  ({done/1e6:.0f}/{total_size/1e6:.0f} MB)", end="", flush=True)
    urllib.request.urlretrieve(VIT_URL, VIT_PATH, reporthook=show_progress)
    print(f"\n완료: {VIT_PATH}")
```

**실제 출력:**
```
가중치 파일 이미 존재: pretrained/ViT-B_16.npz  (413 MB)
```

**코드 설명:**
- `PRETRAINED_DIR.mkdir(exist_ok=True)`: 폴더 만들기 (이미 있으면 무시)
- `VIT_URL`: Google 공식 ViT 가중치 URL (ImageNet-21k 사전 훈련)
- `.npz` 형식: numpy 배열을 압축 저장한 파일
- 진행률 콜백 등록 후 다운로드

**왜 이렇게 하는가?**
- **전이학습(Transfer Learning)**: 처음부터 학습하면 수억 장의 이미지가 필요하지만, 미리 훈련된 가중치를 가져오면 작은 데이터(11,788장)만으로도 좋은 성능을 낼 수 있다.
- 비유: 영어 잘하는 사람이 한국어를 배우는 것이 처음부터 한국어를 배우는 것보다 압도적으로 빠르다.

---

### Cell 5 — 모듈 임포트 및 경로 설정

```
[프로젝트 구조]
 ─────────────
 AiffelThon01/
   ├── TransFG/              ← 논문 구현 (sys.path에 추가)
   │   └── models/
   ├── data_loader.py        ← 직접 만든 모듈
   ├── trainer.py
   ├── inference_utils.py
   └── visualization.py
```

```python
import sys, os
from pathlib import Path

TRANSFG_ROOT = Path("TransFG").resolve()
if str(TRANSFG_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFG_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.modeling import VisionTransformer, CONFIGS
from dataset_cub_fixed import CUBDataset
from data_loader       import get_cub_loaders
from trainer           import train, validate
from inference_utils   import predict_single, evaluate_dataset, predict_batch
from visualization     import (
    show_sample_grid, plot_history,
    visualize_predictions, visualize_attention
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT    = Path("data/CUB_200_2011")
PRETRAINED   = Path("pretrained/ViT-B_16.npz")
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Device     : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"DATA_ROOT  : {DATA_ROOT.resolve()}")
print(f"PRETRAINED : {PRETRAINED.resolve()}")
```

**실제 출력:**
```
Device     : cuda
GPU        : NVIDIA GB10
DATA_ROOT  : /home/changilkim/Documents/aiffel_class/AiffelThon01/data/CUB_200_2011
PRETRAINED : /home/changilkim/Documents/aiffel_class/AiffelThon01/pretrained/ViT-B_16.npz
```

**코드 설명 (핵심):**
- `sys.path.insert(0, ...)`: 파이썬 import 검색 경로 맨 앞에 `TransFG/` 추가 → `from models.modeling import ...` 가능해짐
- `from models.modeling import VisionTransformer, CONFIGS`: 논문 구현체에서 모델 클래스와 설정 불러오기
- `from dataset_cub_fixed import CUBDataset`: CUB-200 dataset wrapper
- `from trainer import train`: 학습 루프 함수
- `DEVICE = torch.device("cuda" if ... else "cpu")`: GPU 우선 사용, 없으면 CPU
- `OUTPUT_DIR.mkdir(exist_ok=True)`: 결과 저장 폴더 미리 생성

**왜 이렇게 하는가?**
- 논문 구현체가 `TransFG/` 폴더 안에 있어서 sys.path에 추가하지 않으면 import 안 된다.
- 경로/설정을 한 곳에 모아두면 나중에 바꾸기 쉽다 (좋은 코딩 습관).

---

### Cell 6 — 데이터셋 탐색

```
[CUB-200-2011 통계]
 ───────────────────
 Train: 5,994장        클래스: 200종
 Test : 5,794장        장당 평균: 30장
 Total: 11,788장
```

```python
from collections import Counter

trainset = CUBDataset(root=str(DATA_ROOT), is_train=True)
testset  = CUBDataset(root=str(DATA_ROOT), is_train=False)
class_names = trainset.get_class_names()

print(f"Train 샘플 수  : {len(trainset):,}")
print(f"Test  샘플 수  : {len(testset):,}")
print(f"클래스 수      : {len(class_names)}")

print("\n클래스 예시:")
for i in [0, 49, 99, 149, 199]:
    print(f"  [{i:3d}] {class_names[i]}")

label_counts = Counter(trainset.labels)
counts = sorted(label_counts.values())
print(f"\n클래스별 Train 샘플: min={counts[0]}, max={counts[-1]}, avg={sum(counts)/len(counts):.1f}")

img, label = trainset[0]
print(f"\n샘플[0] 타입: {type(img)}, label={label} ({class_names[label]})")
```

**실제 출력:**
```
Train 샘플 수  : 5,994
Test  샘플 수  : 5,794
클래스 수      : 200

클래스 예시:
  [  0] 001.Black_footed_Albatross
  [ 49] 050.Eared_Grebe
  [ 99] 100.Brown_Pelican
  [149] 150.Sage_Thrasher
  [199] 200.Common_Yellowthroat

클래스별 Train 샘플: min=29, max=30, avg=30.0
샘플[0] 타입: <class 'PIL.Image.Image'>, label=0 (001.Black_footed_Albatross)
```

**코드 설명:**
- `CUBDataset(root=..., is_train=True/False)`: train/test split 분리
- `get_class_names()`: 200개 클래스 이름 리스트
- `Counter(trainset.labels)`: 각 클래스별 샘플 개수 세기
- `trainset[0]`: 첫 번째 샘플 → `(이미지, 라벨)` 튜플 반환
- 샘플 이미지는 아직 PIL.Image (텐서로 변환 전)

**왜 이렇게 하는가?**
- **데이터 이해는 모델링의 첫 걸음** — 클래스 불균형(어떤 클래스만 너무 많거나 적으면 학습이 한쪽으로 치우침)을 미리 확인.
- CUB-200은 거의 균등(min=29, max=30)이라 추가 처리 불필요.

---

### Cell 7 — 샘플 이미지 시각화

```
[시각화 흐름]
 ───────────
 PIL.Image  →  Resize(224×224)  →  Tensor  →  Normalize  →  matplotlib
```

```python
from torchvision import transforms
from PIL import Image as PILImage

try:
    _bilinear = PILImage.Resampling.BILINEAR
except AttributeError:
    _bilinear = PILImage.BILINEAR

vis_transform = transforms.Compose([
    transforms.Resize((224, 224), _bilinear),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
vis_trainset = CUBDataset(root=str(DATA_ROOT), is_train=True, transform=vis_transform)

show_sample_grid(vis_trainset, class_names=class_names, n=12, cols=4)
```

**실제 출력:** 12장의 새 이미지가 4열 그리드로 표시되며, 각 이미지 위에 클래스명 표시.

**코드 설명:**
- `transforms.Compose([...])`: 여러 변환을 파이프라인으로 연결
- `transforms.Resize((224,224), BILINEAR)`: 이미지를 224×224 크기로 리사이즈 (양선형 보간)
- `transforms.ToTensor()`: PIL.Image → torch.Tensor (값 범위 [0,255] → [0,1])
- `transforms.Normalize(mean, std)`: ImageNet 평균/표준편차로 정규화
- `try/except`: PIL 버전 호환성 처리 (구버전은 `PILImage.BILINEAR`, 신버전은 `Resampling.BILINEAR`)
- `show_sample_grid`: 직접 만든 시각화 함수

**왜 이렇게 하는가?**
- ImageNet pretrained 모델은 ImageNet 통계로 정규화된 입력을 기대 → 같은 mean/std 사용해야 함.
- 시각화는 학습 전 데이터가 잘 들어오는지 sanity check 하는 가장 좋은 방법.

---

### Cell 8 — 하이퍼파라미터 설정

```
[중요 하이퍼파라미터]
 ──────────────────
 img_size: 448  ← TransFG 표준 (큰 이미지 = 미세 특징 보존)
 batch  : 8    ← GPU 메모리 한계
 lr     : 0.03 ← SGD 기준 큰 값
 steps  : 10000 (full) / 200 (quick)
```

```python
CFG = {
    "model_type"  : "ViT-B_16",
    "split"       : "overlap",
    "slide_step"  : 12,
    "img_size"    : 448,
    "train_batch" : 8,
    "eval_batch"  : 8,
    "num_steps"   : 10000,
    "quick_steps" : 200,
    "lr"          : 3e-2,
    "weight_decay": 0.0,
    "warmup_steps": 500,
    "decay_type"  : "cosine",
    "smoothing"   : 0.0,
    "fp16"        : True,
    "run_name"    : "transfg_cub",
    "num_classes" : 200,
    "num_workers" : 4,
}
print("=== TransFG Config ===")
for k, v in CFG.items():
    print(f"  {k:20s}: {v}")
```

**실제 출력:** 위 딕셔너리의 모든 키-값이 정렬되어 출력됨.

**코드 설명 (핵심):**
- `"split": "overlap"`: 패치를 겹치게 자르는 모드 (TransFG 핵심)
- `"slide_step": 12`: 패치 간격 12 픽셀 (16 패치 크기보다 작으므로 4 픽셀 겹침)
- `"img_size": 448`: 224 대신 448 사용 → 4배 많은 패치
- `"warmup_steps": 500`: 처음 500 step은 lr을 0에서 점점 올림
- `"decay_type": "cosine"`: 그 후 cosine 곡선처럼 lr 감소
- `"fp16": True`: 16-bit 혼합 정밀도 학습
- `"num_workers": 4`: DataLoader가 데이터 로딩에 사용할 프로세스 수

**왜 이렇게 하는가?**

| 설정 | 이유 |
|------|------|
| img_size=448 | 작은 새 부위(부리, 눈)도 잘 보이게 |
| batch=8 | 448×448 큰 이미지 + GPU 메모리 한계 |
| lr=3e-2 | TransFG 논문 표준 값 (SGD 기준) |
| warmup | 학습 초기 불안정 방지 |
| cosine decay | 끝에서 천천히 수렴하도록 |
| fp16 | 메모리 절약 + 속도 향상 |

---

### Cell 9 — 데이터로더 생성

```
[DataLoader 흐름]
 ───────────────
 Dataset         DataLoader               Model
   │             ┌──────────┐
   ├─ img1   →  │ batch    │  → [8,3,448,448]  → forward
   ├─ img2      │ shuffle  │
   ├─ ...       │ workers=4│
   └─ imgN      └──────────┘
```

```python
train_loader, test_loader, trainset, testset = get_cub_loaders(
    data_root        = str(DATA_ROOT),
    train_batch_size = CFG["train_batch"],
    eval_batch_size  = CFG["eval_batch"],
    img_size         = CFG["img_size"],
    num_workers      = CFG["num_workers"],
)

x_batch, y_batch = next(iter(train_loader))
print(f"Input batch  : {x_batch.shape}   dtype={x_batch.dtype}")
print(f"Label batch  : {y_batch.shape}   dtype={y_batch.dtype}")
print(f"Label range  : {y_batch.min().item()} ~ {y_batch.max().item()}")
print(f"Train batches: {len(train_loader)} / epoch")
print(f"Test  batches: {len(test_loader)} / epoch")
print(f"\n메모리 estimte (448×448 FP32): {x_batch.numel() * 4 / 1e6:.1f} MB / batch")
```

**실제 출력:**
```
Input batch  : torch.Size([8, 3, 448, 448])   dtype=torch.float32
Label batch  : torch.Size([8])   dtype=torch.int64
Label range  : 20 ~ 189
Train batches: 749 / epoch
Test  batches: 725 / epoch

메모리 estimte (448×448 FP32): 19.3 MB / batch
```

**코드 설명:**
- `get_cub_loaders(...)`: train/test DataLoader 둘 다 만드는 헬퍼 함수
- `next(iter(train_loader))`: 첫 번째 배치 하나만 꺼내서 shape 확인
- `[8, 3, 448, 448]` = (batch=8, channel=3 RGB, height=448, width=448)
- `int64` 라벨: 클래스 인덱스는 정수
- `x_batch.numel() * 4`: tensor 원소 수 × 4 byte = FP32 배치 메모리

**왜 이렇게 하는가?**
- shape 확인은 모델 입력이 잘 들어오는지 검증하는 필수 단계.
- 5,994 train / 8 batch ≈ 749 batch가 1 epoch.
- 메모리 추정으로 GPU OOM(out of memory)을 미리 방지.

---

### Cell 10 — 모델 생성

```
[모델 생성 흐름]
 ──────────────
 CONFIGS["ViT-B_16"]            ┌── overlap split
        ↓                       ├── slide_step=12
 config 수정         →           │
        ↓                       │
 VisionTransformer 생성    ──── num_classes=200, smoothing=0
        ↓
 ViT-B_16.npz 로드        ──── ImageNet-21k pretrained
        ↓
 .to(DEVICE)              ──── GPU로 이동
```

```python
assert PRETRAINED.exists(), f"가중치 파일 없음: {PRETRAINED}"

config = CONFIGS[CFG["model_type"]]
config.split      = CFG["split"]
config.slide_step = CFG["slide_step"]

print("─ ViT-B/16 Config ─")
print(f"  hidden_size : {config.hidden_size}")
print(f"  num_layers  : {config.transformer['num_layers']}")
print(f"  num_heads   : {config.transformer['num_heads']}")
print(f"  patch_size  : {config.patches['size']}")
print(f"  split       : {config.split}, slide_step={config.slide_step}")

stride = config.slide_step
p      = config.patches["size"][0]
n_patches_side = (CFG["img_size"] - p) // stride + 1
print(f"  patch grid  : {n_patches_side}×{n_patches_side} = {n_patches_side**2} patches")

model = VisionTransformer(
    config,
    img_size        = CFG["img_size"],
    zero_head       = True,
    num_classes     = CFG["num_classes"],
    smoothing_value = CFG["smoothing"],
)

model.load_from(np.load(str(PRETRAINED)))
model = model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n학습 파라미터 수: {n_params/1e6:.1f}M")
print(f"모델 device    : {next(model.parameters()).device}")
```

**실제 출력:**
```
─ ViT-B/16 Config ─
  hidden_size : 768
  num_layers  : 12
  num_heads   : 12
  patch_size  : (16, 16)
  split       : overlap, slide_step=12
  patch grid  : 37×37 = 1369 patches
load_pretrained: grid-size from 14 to 37

학습 파라미터 수: 86.9M
모델 device    : cuda:0
```

**코드 설명 (핵심):**
- `assert PRETRAINED.exists()`: 가중치 없으면 에러 (실수 방지)
- `config = CONFIGS["ViT-B_16"]`: ViT-B/16 표준 설정 가져오기
- `config.split = "overlap"`: 일반 ViT가 아닌 overlap 모드로 변경
- `n_patches_side = (448-16)//12 + 1 = 37`: 가로/세로 37개 → 37×37=1,369 patches
- `zero_head=True`: 분류 헤드는 0으로 초기화 (200 클래스용으로 새로 학습)
- `load_from(np.load(...))`: ImageNet 가중치 로드. 224 → 448 변경에 맞춰 position embedding 자동 보간
- `n_params/1e6 = 86.9M`: 8,690만 개 파라미터

**왜 이렇게 하는가?**
- **patch grid 14 → 37**: 224×224 → 448×448로 이미지를 키우고 stride도 16 → 12로 줄여서 패치 수 폭증.
- **zero_head**: ImageNet은 21,000 클래스인데 우리는 200 클래스 → 출력층은 새로 학습해야 함.
- 86.9M 파라미터는 큰 모델이지만 ViT-B 기준으로는 표준.

---

### Cell 11 — 모델 구조 탐색

```
[VisionTransformer 내부]
 ─────────────────────
 transformer
   └── encoder
        ├── layer[0]   ┐
        ├── layer[1]    │  11개 일반 Block
        ├── ...         │  (attention 누적용)
        ├── layer[10]   ┘
        ├── part_select  ← Part Attention (PSM)
        ├── part_layer   ← 마지막 Block (선택된 patch만 처리)
        └── part_norm    ← 최종 LayerNorm
 part_head             ← Linear (200 클래스 출력)
```

```python
print("=== VisionTransformer 최상위 모듈 ===")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters()) / 1e6
    print(f"  {name:20s}: {type(module).__name__:25s}  ({params:.1f}M params)")

print("\n=== Encoder 내부 ===")
enc = model.transformer.encoder
print(f"  transformer blocks (layer): {len(enc.layer)}개  (Block L0~L{len(enc.layer)-1})")
print(f"  Part_Attention (part_select): {type(enc.part_select).__name__}")
print(f"  Final Block    (part_layer) : {type(enc.part_layer).__name__}")
print(f"  Norm           (part_norm)  : {type(enc.part_norm).__name__}")

print("\n=== 분류 헤드 ===")
print(f"  part_head: {model.part_head}")

model.eval()
dummy = torch.randn(2, 3, CFG["img_size"], CFG["img_size"]).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f"\n추론 출력 shape: {out.shape}  (배치=2, 클래스=200)")
```

**실제 출력:**
```
=== VisionTransformer 최상위 모듈 ===
  transformer         : Transformer                (86.7M params)
  part_head           : Linear                     (0.2M params)
=== Encoder 내부 ===
  transformer blocks (layer): 11개  (Block L0~L10)
  Part_Attention (part_select): Part_Attention
  Final Block    (part_layer) : Block
  Norm           (part_norm)  : LayerNorm
=== 분류 헤드 ===
  part_head: Linear(in_features=768, out_features=200, bias=True)
추론 출력 shape: torch.Size([2, 200])  (배치=2, 클래스=200)
```

**코드 설명 (핵심):**
- `model.named_children()`: 모델의 직접 자식 모듈을 (이름, 모듈) 튜플로 순회
- `sum(p.numel() for p in module.parameters())`: 모듈 내 전체 파라미터 수
- `model.transformer.encoder`: encoder 안으로 더 들어가서 PSM 구조 확인
- `model.eval()`: 추론 모드 (BatchNorm/Dropout 비활성화)
- `torch.randn(2, 3, 448, 448)`: 더미 입력 (랜덤)
- `with torch.no_grad()`: gradient 계산 끄기 (메모리 절약)
- 출력 shape `[2, 200]`: 배치 2개, 200 클래스 logit

**왜 이렇게 하는가?**
- 모델 구조를 직접 확인하면 디버깅 능력이 폭발적으로 향상된다.
- 더미 입력으로 forward 한 번 돌려보면 모델이 잘 동작하는지 즉시 검증 가능.
- TransFG 핵심: 일반 12-layer 중 **마지막 11층까지는 일반 Block**, 그 다음 **PSM**, 그 다음 **마지막 Block(part_layer)** 구조.

---

### Cell 12 — 빠른 학습 테스트 (Quick Training, 200 steps)

```
[학습 루프]
 ─────────
 for step in range(num_steps):
     batch ← train_loader
     loss  ← forward
     backward
     optimizer.step()
     if step % eval_every == 0:
         val_acc ← validate(model, test_loader)
         save best checkpoint
```

```python
import time

print(f"Quick training: {CFG['quick_steps']} steps on {DEVICE}")
print(f"FP16 (AMP)    : {CFG['fp16']}")

start = time.time()
best_acc, history = train(
    model        = model,
    train_loader = train_loader,
    test_loader  = test_loader,
    device       = DEVICE,
    num_steps    = CFG["quick_steps"],
    learning_rate= CFG["lr"],
    warmup_steps = 50,
    decay_type   = CFG["decay_type"],
    eval_every   = 50,
    output_dir   = str(OUTPUT_DIR),
    run_name     = CFG["run_name"],
    fp16         = CFG["fp16"],
    gradient_accumulation_steps = 1,
    max_grad_norm = 1.0,
)
elapsed = time.time() - start
print(f"\n경과 시간: {elapsed:.1f}초")
```

**실제 출력 예시:**
```
Quick training: 200 steps on cuda
FP16 (AMP)    : True
[Step   50/200] Train Loss: 5.2xxx | Val Acc: xx.xx% | Val Loss: 5.xxxx
[Step  100/200] Train Loss: 5.1xxx | Val Acc: xx.xx% | Val Loss: 5.xxxx
[Step  150/200] Train Loss: 5.0xxx | Val Acc: xx.xx% | Val Loss: 5.xxxx
[Step  200/200] Train Loss: 4.9xxx | Val Acc: xx.xx% | Val Loss: 5.xxxx
Best Val Acc: xx.xx%
경과 시간: xxxx.x초
```

**코드 설명 (핵심):**
- `train(...)`: 학습 함수 (return: best_acc, history dict)
- `num_steps=200`: full 학습이 너무 길어서 우선 200 step만 (코드가 잘 도는지 sanity check)
- `warmup_steps=50`: 처음 50 step은 lr을 0 → 0.03으로 점점 증가
- `eval_every=50`: 50 step마다 test set 평가
- `fp16=True`: AMP(Automatic Mixed Precision) 사용
- `max_grad_norm=1.0`: gradient 폭발 방지 (gradient clipping)
- `gradient_accumulation_steps=1`: 실효 batch size = 8 × 1 = 8

**왜 이렇게 하는가?**
- **Quick training은 sanity check** — 코드가 망가지지 않았는지, GPU 메모리가 충분한지 확인하는 용도.
- 200 step으론 절대 좋은 성능 안 나옴 (Val Acc 매우 낮음). 그래도 loss가 줄어드는 추세를 봐야 함.
- gradient clipping(max_grad_norm=1.0)은 ViT 학습에 거의 필수.

---

### Cell 13 — 학습 history 시각화

```
[그래프]
 ──────
 Loss                Val Accuracy
 ┌─────────┐        ┌─────────┐
 │\        │        │       /│
 │ \       │        │      / │
 │  \___   │        │   __/  │
 └─────────┘        └─────────┘
   step                step
```

```python
plot_history(history, eval_every=50)
```

**실제 출력:** Train Loss와 Val Accuracy의 step별 꺾은선 그래프.

**코드 설명:**
- `plot_history`: 직접 만든 함수 (visualization.py에 정의)
- `history`: `{"train_loss": [...], "val_acc": [...], "val_loss": [...]}` 형태의 dict

**왜 이렇게 하는가?**
- 숫자만 봐서는 학습 추세를 파악하기 어렵다 → 시각화 필수.
- Train loss는 줄지만 Val acc가 안 오르면 → 과적합(overfitting)이거나 학습률 문제.

---

### Cell 14 — 전체 학습 (주석 처리됨)

```
[전체 학습 vs Quick]
 ─────────────────
 Quick : 200 step  →   ~몇 분        (sanity check)
 Full  : 10000 step → ~8~12시간      (논문 재현)
```

```python
# 전체 학습 — 주석 해제 후 실행 (~8~12시간)
# model.load_from(np.load(str(PRETRAINED)))
# model = model.to(DEVICE)
# best_acc, history = train(
#     model=model, train_loader=train_loader, test_loader=test_loader,
#     device=DEVICE, num_steps=CFG["num_steps"], ...
# )
# plot_history(history, eval_every=100)
print("전체 학습: 위 주석 해제 후 실행")
print("TensorBoard: tensorboard --logdir logs")
```

**실제 출력:**
```
전체 학습: 위 주석 해제 후 실행
TensorBoard: tensorboard --logdir logs
```

**코드 설명:**
- 일부러 주석 처리 — 무심코 8시간짜리 학습을 시작하면 안 되니까.
- TensorBoard: `tensorboard --logdir logs` 명령으로 학습 진행을 웹 브라우저로 모니터링 가능.

**왜 이렇게 하는가?**
- 노트북 코드는 **위에서 아래로 한 번에 실행될 수 있어야 함**이 원칙. 그러나 8~12시간 걸리는 학습은 예외.
- 전체 학습이 필요한 사용자만 주석 해제해서 실행하도록 설계.

---

### Cell 15 — 체크포인트 로드

```
[체크포인트 시스템]
 ─────────────────
 학습 중 best Val Acc 달성 시
        ↓
 model.state_dict() → output/transfg_cub_checkpoint.bin
        ↓
 추론 시 다시 로드
```

```python
ckpt_path = OUTPUT_DIR / f"{CFG['run_name']}_checkpoint.bin"
if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"체크포인트 로드 완료: {ckpt_path}")
else:
    print(f"체크포인트 없음: {ckpt_path}")
    model.eval()
```

**실제 출력:**
```
체크포인트 로드 완료: output/transfg_cub_checkpoint.bin
```

**코드 설명:**
- `torch.load(..., map_location=DEVICE)`: 체크포인트를 GPU/CPU 어디로 로드할지 지정
- `ckpt["model"]`: 저장 시 dict 형태로 했음 → `{"model": state_dict, "step": ..., ...}`
- `model.load_state_dict(...)`: 모델 파라미터에 가중치 덮어쓰기
- `model.eval()`: 추론 모드 (BatchNorm/Dropout 끄기)

**왜 이렇게 하는가?**
- 학습은 시간이 걸리므로 best 결과를 디스크에 저장 → 추론 시 재사용.
- 가장 잘 했던 시점의 모델로 평가하는 것이 표준.

---

### Cell 16 — 테스트셋 평가

```
[전체 테스트셋 평가]
 ──────────────────
 5,794장 → forward → softmax → argmax → 정답 비교
                                            ↓
                                       accuracy %
```

```python
print("테스트셋 평가 중...")
result = evaluate_dataset(model, test_loader, DEVICE)
print(f"\n=== 평가 결과 ===")
print(f"Test Accuracy : {result['accuracy']:.4f}  ({result['accuracy']*100:.2f}%)")
print(f"Test Avg Loss : {result['avg_loss']:.5f}")
print(f"\n※ TransFG 논문 최고 성능 (CUB-200-2011)")
print(f"   overlap split, 10k steps: 91.7%")
```

**실제 출력:**
```
=== 평가 결과 ===
Test Accuracy : 0.3788  (37.88%)
Test Avg Loss : 4.33435
※ TransFG 논문 최고 성능 (CUB-200-2011)
   overlap split, 10k steps: 91.7%
```

**코드 설명:**
- `evaluate_dataset`: test_loader 전체를 돌며 accuracy 계산
- `result["accuracy"]`: 0.0 ~ 1.0 범위
- `result["avg_loss"]`: cross-entropy 평균

**왜 이렇게 하는가?**
- 200 step Quick 학습으로는 37.88% (논문은 91.7%) — 정상 결과.
- 무작위 추측이면 1/200 = 0.5% 이므로 37%도 학습은 일어났음을 의미.
- 91.7%까지 가려면 full 10,000 step 학습 필요.

**Quick vs Full 학습 비교:**

| 항목 | Quick (200 steps) | Full (10,000 steps) |
|------|------------------|---------------------|
| 시간 | 수 분             | 8~12시간            |
| Val Acc | 30~40%        | 90% 이상           |
| 목적 | 코드 sanity check | 논문 재현           |

---

### Cell 17 — 단일 이미지 추론

```
[단일 추론 흐름]
 ──────────────
 이미지 1장 → preprocess → model → softmax
                                     │
                                     ├── argmax → 1순위 예측
                                     └── topk(5) → Top-5 예측
```

```python
import random
idx = random.randint(0, len(testset) - 1)
img_path   = testset.img_paths[idx]
true_label = testset.labels[idx]

result = predict_single(model, img_path, DEVICE, class_names=class_names)
print(f"이미지 경로 : {img_path}")
print(f"실제 클래스 : [{true_label:3d}] {class_names[true_label]}")
print(f"예측 클래스 : [{result['pred_idx']:3d}] {result.get('pred_name', '')}")
print(f"신뢰도      : {result['confidence']:.4f}")
correct = result['pred_idx'] == true_label
print(f"정답 여부   : {'✓ 정답' if correct else '✗ 오답'}")
print("\nTop-5 예측:")
for name, prob in result.get("top5", []):
    bar = "█" * int(prob * 40)
    short_name = name.split(".")[-1].strip()
    print(f"  {prob:.4f} {bar} {short_name}")
```

**실제 출력 예시:**
```
이미지 경로 : data/CUB_200_2011/images/161.Blue_winged_Warbler/Blue_Winged_Warbler_0078.jpg
실제 클래스 : [160] 161.Blue_winged_Warbler
예측 클래스 : [176] 177.Prothonotary_Warbler
신뢰도      : 0.0304
정답 여부   : ✗ 오답
Top-5 예측:
  0.0304 █ Prothonotary_Warbler
  0.0199  Yellow_Warbler
  ...
```

**코드 설명:**
- `random.randint(0, len(testset)-1)`: 테스트 셋에서 랜덤 인덱스
- `predict_single(...)`: 이미지 경로 받아서 예측
- `result["pred_idx"]`: 예측 클래스 인덱스 (0~199)
- `result["confidence"]`: softmax 최대값 (0~1)
- `result["top5"]`: 상위 5개 (이름, 확률) 리스트
- `bar = "█" * int(prob * 40)`: 확률을 막대 바로 시각화
- `name.split(".")[-1]`: "001.Black_footed_Albatross" → "Black_footed_Albatross"

**왜 이렇게 하는가?**
- Top-5: Top-1만 보면 모델이 헷갈리는 정도를 모름. **두 종이 거의 비슷한 확률**이면 모델이 어려워한다는 신호.
- Quick 학습 모델이라 신뢰도가 매우 낮음(0.0304) — 정상.

---

### Cell 18 — 배치 예측 시각화

```
[배치 시각화]
 ───────────
 8장 이미지 그리드, 정답 = 초록 테두리, 오답 = 빨강 테두리
 ┌────┬────┬────┬────┐
 │ 정 │ 오 │ 정 │ 정 │
 ├────┼────┼────┼────┤
 │ 오 │ 정 │ 정 │ 오 │
 └────┴────┴────┴────┘
```

```python
x_vis, y_vis = next(iter(test_loader))
preds, probs = predict_batch(model, x_vis, DEVICE)
batch_acc = (preds == y_vis.numpy()).mean()
print(f"배치 정확도: {batch_acc:.4f}")
visualize_predictions(
    x_vis[:8], preds[:8], y_vis.numpy()[:8],
    class_names=class_names, cols=4
)
```

**실제 출력:** 8장 이미지 4×2 그리드. 각 이미지 위에 "예측: XXX (정답/오답)".

**코드 설명:**
- `next(iter(test_loader))`: 첫 번째 test 배치 (8장)
- `predict_batch(...)`: 배치 단위 예측 → (preds, probs)
- `(preds == y_vis.numpy()).mean()`: True/False를 0/1로 → 평균 = 정확도
- `visualize_predictions`: 이미지 그리드 + 예측 결과 표시

**왜 이렇게 하는가?**
- 숫자(accuracy)만 보면 모델이 어떤 종류의 이미지를 잘하는지/못하는지 알 수 없음.
- 시각적으로 확인하면 "이 모델은 작은 새는 잘 맞추는데 비슷한 색깔 새는 못 맞춘다" 같은 통찰 가능.

---

### Cell 19 — Self-Attention Map 시각화

```
[Attention 시각화]
 ───────────────
 입력 이미지 → Layer별 attention map 추출
        ↓                            ↓
 [원본 이미지]   →  [L0]  [L1]  [L2] ... [L11]  (히트맵 오버레이)
                  넓게      점점       부리/눈에
                  본다       좁아짐    집중
```

```python
from data_loader import get_cub_transforms
_, test_tf = get_cub_transforms(CFG["img_size"])
attn_testset = CUBDataset(str(DATA_ROOT), is_train=False, transform=test_tf)

idx = 0
img_tensor, label = attn_testset[idx]
print(f"클래스: [{label}] {class_names[label]}")
print("레이어별 CLS→Patch attention map (head=0)")

visualize_attention(
    model, img_tensor, DEVICE,
    head=0, patch_size=16, img_size=CFG["img_size"]
)
```

**실제 출력:** 12개 attention 히트맵 (Layer 0 ~ 11) — 각 레이어가 이미지의 어디에 주목하는지 시각화.

**코드 설명:**
- `get_cub_transforms(img_size)`: train/test transform pair 반환 → test 변환만 사용
- `attn_testset[0]`: 0번 이미지 (이미 텐서 변환됨)
- `visualize_attention(model, img, ..., head=0, patch_size=16)`:
  - CLS 토큰이 각 patch를 얼마나 보는지를 layer마다 시각화
  - head=0: 12개 attention head 중 0번
  - patch_size=16: ViT-B/16의 패치 크기

**왜 이렇게 하는가?**
- **Attention map은 모델의 "눈"** — 어디를 보고 결정하는지 보여줌.
- **초기 레이어**: 전체적인 윤곽 파악 (넓게 봄)
- **후기 레이어**: 부리, 눈 같은 미세 부위에 집중 (좁게 봄) — TransFG의 핵심 가설.

---

### Cell 20 — 12개 Head Attention 비교

```
[Multi-Head Attention]
 ───────────────────
 마지막 Block 안의 12개 head는 각자 다른 곳을 본다
 ┌────┬────┬────┬────┐
 │ H0 │ H1 │ H2 │ H3 │  ← head 0~3 (예: 머리)
 ├────┼────┼────┼────┤
 │ H4 │ H5 │ H6 │ H7 │  ← head 4~7 (예: 날개)
 ├────┼────┼────┼────┤
 │ H8 │ H9 │H10 │H11 │  ← head 8~11 (예: 꼬리, 발)
 └────┴────┴────┴────┘
```

```python
from visualization import AttentionMapExtractor
import numpy as np

idx = 10
img_tensor, label = attn_testset[idx]
x = img_tensor.unsqueeze(0).to(DEVICE)

model.eval()
with AttentionMapExtractor(model) as ext:
    with torch.no_grad():
        _ = model(x)
    attn_maps = ext.attention_maps  # list[(1, 12, seq, seq)]

if attn_maps:
    last = attn_maps[-1][0]  # (12, seq, seq) — 마지막 Block
    n_heads = last.shape[0]
    cls_attn = last[:, 0, 1:].numpy()  # (12, n_patches)
    n_p = int(cls_attn.shape[1] ** 0.5)

    if n_p * n_p == cls_attn.shape[1]:
        from PIL import Image as PILImage
        from visualization import denormalize
        img_rgb = denormalize(img_tensor)
        cols = 4
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten()
        for h in range(n_heads):
            am = cls_attn[h].reshape(n_p, n_p)
            am = (am - am.min()) / (am.max() - am.min() + 1e-8)
            try:
                resample = PILImage.Resampling.BILINEAR
            except AttributeError:
                resample = PILImage.BILINEAR
            am_up = np.array(
                PILImage.fromarray((am*255).astype(np.uint8))
                .resize((CFG["img_size"], CFG["img_size"]), resample)
            ) / 255.0
            axes[h].imshow(img_rgb)
            axes[h].imshow(am_up, alpha=0.5, cmap="jet")
            axes[h].set_title(f"Head {h}", fontsize=8)
            axes[h].axis("off")
        for ax in axes[n_heads:]:
            ax.axis("off")
        plt.suptitle(f"마지막 Block 12개 Head — {class_names[label].split('.')[-1]}", fontsize=11)
        plt.tight_layout()
        plt.show()
    else:
        print(f"overlap split: patch 수({cls_attn.shape[1]})가 정방형이 아닙니다.")
else:
    print("Attention map을 추출하지 못했습니다.")
```

**실제 출력:** 3행 4열 그리드 — 12개 head별 attention 히트맵.

**코드 설명 (핵심):**
- `AttentionMapExtractor(model)`: context manager — 들어갈 때 hook 등록, 나올 때 제거
- `with ext: ... ext.attention_maps`: forward pass 동안 모든 layer의 attention 저장
- `attn_maps[-1]`: 마지막 Block (PSM 통과 후)
- `last[:, 0, 1:]`: `(head, query=0=CLS, key=1: 모든 patch)` → CLS→patch attention
- `am.reshape(n_p, n_p)`: 1D → 2D 격자 (overlap이 아니면 정방형)
- `(am - am.min()) / (am.max() - am.min())`: 0~1 정규화
- `axes[h].imshow(img_rgb)` 후 `imshow(am_up, alpha=0.5, cmap="jet")`: 원본 위에 히트맵 50% 투명 오버레이
- `axes[h].axis("off")`: 좌표축 숨김

**왜 이렇게 하는가?**
- **각 head는 독립적인 "관찰자"** — 같은 이미지를 다르게 본다.
- 어떤 head는 부리, 어떤 head는 눈, 어떤 head는 날개에 주목.
- Multi-Head Attention의 강력함을 시각적으로 이해 가능.
- `overlap=True`이면 patch 수가 1369 = 37×37 정사각형이라 reshape 가능. 일반 split이면 불가능.

---

## 5. 전체 셀 요약 표

| 셀 | 단계 | 내용 | 핵심 함수/객체 |
|-----|------|------|---------------|
| Cell 0 | 준비 | Python 인터프리터 확인 | `sys.executable` |
| Cell 1 | 준비 | GPU/CUDA 환경 확인 | `torch.cuda.is_available()` |
| Cell 2 | 준비 | 패키지 설치 | (이미 venv에 설치됨) |
| Cell 3 | 데이터 | CUB-200-2011 다운로드 | `urllib.request.urlretrieve` |
| Cell 4 | 데이터 | ViT-B_16 가중치 다운로드 | `urllib.request.urlretrieve` |
| Cell 5 | 임포트 | 모듈 import, 경로 설정 | `sys.path.insert` |
| Cell 6 | 데이터 | 데이터셋 통계 탐색 | `CUBDataset`, `Counter` |
| Cell 7 | 데이터 | 샘플 이미지 그리드 | `transforms.Compose` |
| Cell 8 | 모델 | 하이퍼파라미터 설정 | `CFG` dict |
| Cell 9 | 모델 | DataLoader 생성 | `get_cub_loaders` |
| Cell 10 | 모델 | 모델 생성 + pretrained | `VisionTransformer`, `model.load_from` |
| Cell 11 | 모델 | 모델 구조 분석 | `model.named_children()` |
| Cell 12 | 학습 | Quick 학습 (200 steps) | `train(...)` |
| Cell 13 | 학습 | history 시각화 | `plot_history` |
| Cell 14 | 학습 | 전체 학습 (주석) | (주석 해제 시) |
| Cell 15 | 추론 | 체크포인트 로드 | `torch.load`, `load_state_dict` |
| Cell 16 | 추론 | 테스트셋 평가 | `evaluate_dataset` |
| Cell 17 | 추론 | 단일 이미지 추론 + Top-5 | `predict_single` |
| Cell 18 | 추론 | 배치 예측 시각화 | `predict_batch`, `visualize_predictions` |
| Cell 19 | 분석 | Self-Attention map | `visualize_attention` |
| Cell 20 | 분석 | 12개 Head 비교 | `AttentionMapExtractor` |

---

## 6. 공부 팁

```
[학습 로드맵]
 ───────────
 Day 1-2  ───  Python 기초 + numpy 익히기
        │
        ▼
 Day 3-5  ───  PyTorch 기본 (Tensor, autograd, nn.Module)
        │
        ▼
 Day 6-8  ───  CNN 기초 (LeNet, ResNet 따라 만들기)
        │
        ▼
 Day 9-11 ───  Transformer 원리 (Attention is All You Need)
        │
        ▼
 Day 12-14 ── ViT 논문 → TransFG 논문 → 이 노트북!
```

### 6-1. 처음 보는 사람은 이 순서로 보자

1. **Cell 1, 5**: 환경/임포트만 먼저 이해
2. **Cell 6, 7**: 데이터를 직접 보고 만져보자
3. **Cell 10, 11**: 모델이 어떻게 생겼는지 파악
4. **Cell 19, 20**: Attention 시각화로 "모델의 눈" 체감
5. **Cell 12, 13**: 그제서야 학습 코드 들여다보기

### 6-2. 자주 막히는 곳

| 막히는 곳 | 원인 | 해결 |
|----------|------|------|
| `CUDA out of memory` | batch가 너무 큼 | batch를 4로 줄이거나 fp16=True |
| `import error` | venv 커널 미선택 | Jupyter 우상단 커널 변경 |
| Val Acc가 1%대 | 학습이 안 일어남 | lr이 너무 큰지/작은지 확인 |
| 진행률 0%에서 멈춤 | 데이터 다운로드 끊김 | 직접 wget으로 다운로드 |
| matplotlib 한글 깨짐 | 폰트 미설정 | `Noto Sans CJK JP` 사용 |

### 6-3. 직접 해보기 좋은 실험

1. **batch size 바꾸기**: 8 → 4 → 16 (메모리/속도 변화 관찰)
2. **img_size 바꾸기**: 448 → 224 (성능 변화 관찰)
3. **Top-5 시각화**: 다른 인덱스의 이미지로 추론
4. **Attention head 비교**: head=0 외 다른 head도 시각화
5. **Quick training step 늘리기**: 200 → 1000 (성능 향상 추세)

### 6-4. 더 깊이 공부하려면

- **ViT 논문**: "An Image is Worth 16×16 Words" (Dosovitskiy et al., 2020)
- **TransFG 논문**: "TransFG: A Transformer Architecture for Fine-Grained Recognition" (He et al., AAAI 2022)
- **Attention 원리**: "The Illustrated Transformer" (Jay Alammar 블로그) ← **꼭 읽기**
- **PyTorch 공식 튜토리얼**: pytorch.org/tutorials

### 6-5. 핵심 비유 다시 정리

| 개념 | 비유 |
|------|------|
| ViT | 사진을 직소 퍼즐로 잘라서 NLP처럼 처리 |
| Patch | 퍼즐 한 조각 (16×16 픽셀) |
| Overlap | 퍼즐을 일부러 겹치게 자름 |
| Attention | "어디를 봐야 할까" 하는 시선 |
| Multi-Head | 12명의 관찰자가 각자 다른 곳을 봄 |
| Pretrained | 영어 잘하는 사람이 한국어 배우기 |
| FP16 | 자(ruler)를 mm → cm 단위로 (약간 거칠지만 빠름) |
| GPU | 수천 명의 알바생이 동시에 단순 계산 |
| Fine-Grained | "푸들 vs 말티즈" 같은 미세 분류 |

### 6-6. 마지막 한 마디

> 코드를 한 줄씩 직접 쳐보고, 출력을 바꿔보고, 일부러 에러를 내보자.
> 에러 메시지를 두려워하지 않는 사람이 가장 빨리 성장한다.
> 모르는 함수는 `?function_name`을 노트북에 입력하면 도움말이 나온다.

---

**작성 완료** — 이 문서는 노트북 `TransFG_CUB.ipynb`의 학습용 가이드이다.
