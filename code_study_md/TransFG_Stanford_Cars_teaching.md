# TransFG로 자동차 196종 분류하기 (Stanford Cars)

> 대학 1학년을 위한 친절한 노트북 해설서  
> 노트북 원본: `TransFG_Stanford_Cars.ipynb`

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [핵심 개념 (비유로 이해하기)](#2-핵심-개념-비유로-이해하기)
3. [전체 구조 (ASCII 다이어그램)](#3-전체-구조-ascii-다이어그램)
4. [CUB vs Stanford Cars 비교](#4-cub-vs-stanford-cars-비교)
5. [Stanford Cars 데이터셋 특화 포인트](#5-stanford-cars-데이터셋-특화-포인트)
6. [셀별 코드 설명](#6-셀별-코드-설명)
7. [공부 팁](#7-공부-팁)

---

## 1. 한 줄 요약

> **TransFG (Vision Transformer 기반 모델)** 로 비슷하게 생긴 자동차 196종을 사진만 보고 구별하는 프로젝트.  
> "이건 2012년 Volvo C30 Hatchback이야 vs 아니야 2007년 Chevrolet Silverado야" 수준의 미세한 차이를 학습한다.

---

## 2. 핵심 개념 (비유로 이해하기)

### 2.1 Fine-Grained Image Classification (미세 분류)

```
일반 분류:        고양이 vs 강아지 vs 자동차    ← 누구나 구별 가능
미세 분류 (FGVC): BMW 3시리즈 2010 vs BMW 3시리즈 2012 ← 전문가도 헷갈림
```

**비유:** 일반 분류가 "동물원에서 동물 이름 맞추기"라면, 미세 분류는 "자동차 동호회에서 모델·연식 맞추기"다. 같은 BMW라도 그릴 모양, 헤드라이트 곡선, 휠 디자인이 살짝 다른데, 사람도 자세히 봐야 안다.

### 2.2 Vision Transformer (ViT)

```
사진 한 장 → [16x16 작은 조각(patch) 1369개로 자름] → 각 조각을 단어처럼 취급
       → Transformer가 "조각들 사이의 관계"를 학습 → 분류
```

**비유:** 자동차 사진을 직소 퍼즐 1369조각으로 잘라서, "이 조각(그릴 부분)과 저 조각(헤드라이트 부분)이 같이 있을 때 → BMW다" 같은 패턴을 학습한다.

### 2.3 TransFG의 핵심 아이디어 — "PSM (Part Selection Module)"

```
일반 ViT:    1369개 조각 모두 평등하게 봄
TransFG:    "중요한 조각만 골라서" 다시 한 번 자세히 봄
            → 그릴, 헤드라이트, 휠 같은 식별 부위에 집중!
```

### 2.4 Overlap Split

```
일반 patch:        ▢▢▢▢      (16x16, 겹치지 않음, 28x28=784개)
overlap (slide=12): ▢▢▢▢      (16x16, 옆 조각과 겹침, 37x37=1369개)
                    ▢▢▢       
                    겹쳐서 더 촘촘히 본다 → 경계 정보 손실 방지
```

**비유:** 책을 읽을 때 "단어 단위"가 아니라 "단어를 약간씩 겹쳐 읽기" — 문맥이 더 잘 이어진다.

---

## 3. 전체 구조 (ASCII 다이어그램)

```
┌──────────────────────────────────────────────────────────────────┐
│                    TransFG Stanford Cars Pipeline                │
└──────────────────────────────────────────────────────────────────┘

[Cell 1]  환경 확인 (Python, PyTorch, CUDA, GPU)
              │
              ▼
[Cell 2]  모듈 임포트 + 경로 설정
              │  TRANSFG_ROOT, DATA_ROOT, PRETRAINED, OUTPUT_DIR
              ▼
[Cell 3]  데이터셋 탐색 (8,144 train / 8,041 test / 196 classes)
              │
              ▼
[Cell 4]  샘플 이미지 시각화 (12개 자동차 그리드)
              │
              ▼
[Cell 5]  하이퍼파라미터 CFG 딕셔너리
              │
              ▼
[Cell 6]  DataLoader 생성 (배치 크기 8, 1018 batches/epoch)
              │
              ▼
[Cell 7]  ViT-B_16 모델 생성 + ImageNet 사전학습 가중치 로드
              │  (분류 헤드는 새로 만듦: 1000 → 196)
              ▼
[Cell 8]  Quick Training (200 steps, ~70분)
              │  └─ 학습이 잘 되는지 빠르게 확인
              ▼
[Cell 9]  학습 곡선 시각화
              │
              ▼
[Cell 10] 전체 학습 (10,000 steps, 주석 처리)
              │
              ▼
[Cell 11] 체크포인트 로드 (있으면)
              │
              ▼
[Cell 12] 테스트셋 전체 평가
              │
              ▼
[Cell 13] 단일 이미지 추론 + Top-5
              │
              ▼
[Cell 14] 배치 예측 시각화 (정답 초록 / 오답 빨강)
              │
              ▼
[Cell 15] Attention Map (CLS → Patch, 레이어별)
              │
              ▼
[Cell 16] 12개 Head Attention 비교
              │
              ▼
[Cell 17] CUB vs Stanford Cars 비교 표
```

---

## 4. CUB vs Stanford Cars 비교

| 항목 | CUB-200-2011 | **Stanford Cars** |
|---|---|---|
| 도메인 | 조류 (새) | **자동차** |
| 클래스 수 | 200종 | **196종** |
| Train 이미지 | 5,994장 | **8,144장** |
| Test 이미지 | 5,794장 | **8,041장** |
| 클래스당 샘플 (평균) | ~30장 | **~41.6장** |
| 논문 SOTA | 91.7% | **96.1%** |
| 핵심 구별 부위 | 부리·깃털·눈 | **그릴·헤드라이트·휠** |
| 배경 다양성 | 높음 (자연환경) | **낮음 (스튜디오/전시장)** |
| 데이터 형식 | 파일시스템 (jpg) | **Parquet (이미지 내장)** |

> **왜 Stanford Cars가 정확도가 더 높은가?**  
> 1) 클래스당 학습 샘플이 더 많음 (41.6 vs 30)  
> 2) 배경이 단순해서 모델이 자동차 자체에 집중하기 쉬움  
> 3) 자동차는 디자인이 인공적·기계적이라 "정형화된 식별 부위"(그릴, 휠)가 명확함

---

## 5. Stanford Cars 데이터셋 특화 포인트

### 5.1 클래스 명명 방식

```
형식: [제조사] [모델명] [차종] [연도]

예시:
  [  0] AM General Hummer SUV 2000
  [ 49] Chevrolet Silverado 1500 Classic Extended Cab 2007
  [ 99] Ford F-450 Super Duty Crew Cab 2012
  [149] Ram C/V Cargo Van Minivan 2012
  [195] Volvo C30 Hatchback 2012
```

> **중요:** 같은 모델이라도 **연식이 다르면 별개 클래스**다. 즉 모델은 "BMW 3시리즈 2010"과 "BMW 3시리즈 2012"를 다른 클래스로 학습해야 한다.

### 5.2 코드 관점에서 CUB와 다른 점

| 구성 | CUB | Stanford Cars |
|---|---|---|
| Dataset 클래스 | `CUBDataset` | `StanfordCarsDataset` |
| DataLoader 함수 | `get_cub_loaders` | `get_cars_loaders` |
| `num_classes` | 200 | **196** |
| 데이터 저장 | 파일시스템 jpg | **parquet 안에 내장** |

---

## 6. 셀별 코드 설명

---

### Cell 1 — 환경 확인

```
┌─────────────────────────────────────┐
│  Python  PyTorch  CUDA  GPU  Memory │
└─────────────────────────────────────┘
```

```python
import sys
import torch

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**실제 출력:**
```
Python : 3.12.3
PyTorch: 2.10.0+cu130
CUDA   : True
GPU    : NVIDIA GB10
Memory : 128.0 GB
```

**코드 설명:**
- `import sys, torch`: 파이썬 시스템 정보와 PyTorch 임포트
- `sys.version.split()[0]`: "3.12.3 (main, ...)" 같은 긴 문자열에서 버전 숫자만 추출
- `torch.__version__`: 설치된 PyTorch 버전 (cu130 = CUDA 13.0 빌드)
- `torch.cuda.is_available()`: GPU 사용 가능 여부 (True여야 학습 빠름)
- `torch.cuda.get_device_name(0)`: 0번 GPU 이름 (NVIDIA GB10 = DGX Spark)
- `total_memory / 1e9`: 바이트를 GB로 변환

> **왜 이렇게 하는가?** 학습 시작 전에 환경이 제대로 잡혔는지 확인하는 안전장치. CUDA가 False면 CPU로 학습되어 100배 느려진다.

---

### Cell 2 — 모듈 임포트 및 경로 설정

```
┌────────────────────────────────────────────────┐
│  TransFG/  ──→  sys.path 추가  ──→  import 가능  │
│                                                │
│  data/Stanford_Cars/  pretrained/  output/     │
└────────────────────────────────────────────────┘
```

```python
import sys, io, random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

TRANSFG_ROOT = Path("TransFG").resolve()
if str(TRANSFG_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFG_ROOT))

from models.modeling import VisionTransformer, CONFIGS
from dataset_stanford_cars import StanfordCarsDataset
from data_loader_cars       import get_cars_loaders
from trainer          import train, validate
from inference_utils  import predict_batch, evaluate_dataset
from visualization    import show_sample_grid, plot_history, visualize_predictions, visualize_attention

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT  = Path("data/Stanford_Cars")
PRETRAINED = Path("pretrained/ViT-B_16.npz")
OUTPUT_DIR = Path("output/stanford_cars")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device    : {DEVICE}")
print(f"DATA_ROOT : {DATA_ROOT.resolve()}")
print(f"PRETRAINED: {PRETRAINED.resolve()}")
assert DATA_ROOT.exists(),  "데이터셋 없음 — data/Stanford_Cars 확인"
assert PRETRAINED.exists(), "가중치 없음 — pretrained/ViT-B_16.npz 확인"
```

**실제 출력:**
```
Device    : cuda
DATA_ROOT : /workspace/data/Stanford_Cars
PRETRAINED: /workspace/pretrained/ViT-B_16.npz
```

**코드 설명 (핵심 위주):**
- `Path("TransFG").resolve()`: 상대경로를 절대경로로 변환 — 어디서 노트북을 실행해도 동작
- `sys.path.insert(0, ...)`: TransFG 폴더를 임포트 경로에 추가 → 그 안의 `models/modeling.py` 등을 모듈처럼 사용
- `from dataset_stanford_cars import StanfordCarsDataset`: **Stanford Cars 전용 Dataset 클래스** (CUB 노트북과 다른 핵심 부분)
- `from data_loader_cars import get_cars_loaders`: **자동차용 DataLoader 함수** (CUB는 `get_cub_loaders`)
- `DEVICE = torch.device("cuda" if ...)`: GPU 있으면 GPU, 없으면 CPU 자동 선택
- `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)`: 출력 폴더 자동 생성 (이미 있어도 에러 안 남)
- `assert ... .exists()`: 데이터셋·가중치 파일이 없으면 즉시 멈춤 (디버깅 시간 절약)

> **왜 이렇게 하는가?** 경로 문제는 ML 프로젝트에서 가장 흔한 에러 원인. 미리 절대경로로 변환하고 `assert`로 검증하면 학습 도중 "파일 없음" 에러로 시간 낭비를 막을 수 있다.

---

### Cell 3 — 데이터셋 탐색

```
┌─────────────────────────────────────────────┐
│  Stanford Cars                              │
│   ├─ Train: 8,144장                         │
│   ├─ Test : 8,041장                         │
│   └─ 196 classes (제조사+모델+연도)         │
└─────────────────────────────────────────────┘
```

```python
from collections import Counter

trainset_raw = StanfordCarsDataset(root=str(DATA_ROOT), split="train")
testset_raw  = StanfordCarsDataset(root=str(DATA_ROOT), split="test")
class_names  = trainset_raw.get_class_names()

print(f"Train 샘플 수 : {len(trainset_raw):,}")
print(f"Test  샘플 수 : {len(testset_raw):,}")
print(f"클래스 수     : {len(class_names)}")

print("\n클래스 예시:")
for i in [0, 49, 99, 149, 195]:
    print(f"  [{i:3d}] {class_names[i]}")

counts = sorted(Counter(trainset_raw.labels).values())
print(f"\n클래스별 Train 샘플: min={counts[0]}, max={counts[-1]}, avg={sum(counts)/len(counts):.1f}")

img, label = trainset_raw[0]
print(f"\n샘플[0]: PIL size={img.size}, label={label} ({class_names[label]})")
```

**실제 출력:**
```
Train 샘플 수 : 8,144
Test  샘플 수 : 8,041
클래스 수     : 196

클래스 예시:
  [  0] AM General Hummer SUV 2000
  [ 49] Chevrolet Silverado 1500 Classic Extended Cab 2007
  [ 99] Ford F-450 Super Duty Crew Cab 2012
  [149] Ram C/V Cargo Van Minivan 2012
  [195] Volvo C30 Hatchback 2012

클래스별 Train 샘플: min=24, max=68, avg=41.6
샘플[0]: PIL size=(400, 300), label=0 (AM General Hummer SUV 2000)
```

**코드 설명 (핵심 위주):**
- `StanfordCarsDataset(root=..., split="train")`: train 또는 test 이미지를 로드하는 Dataset 객체 생성. `transform=None`이면 PIL 이미지 그대로 반환
- `class_names`: 196개 클래스 이름 리스트 (인덱스 → 사람이 읽을 이름)
- `Counter(trainset_raw.labels)`: 각 클래스에 몇 장씩 있는지 카운트 → 클래스 불균형 확인
- `min=24, max=68, avg=41.6`: 가장 적은 클래스도 24장, 가장 많은 클래스는 68장 → 비교적 균형 잡힘
- `trainset_raw[0]`: 첫 번째 샘플을 (PIL Image, label) 튜플로 반환
- `PIL size=(400, 300)`: 이미지마다 원본 크기가 다름 → 나중에 transform으로 통일

> **왜 이렇게 하는가?** 학습 전에 "데이터가 정말 잘 들어왔나?"를 눈으로 확인하는 것이 매우 중요. 클래스 불균형이 심하면 별도 처리(weighted sampling 등)가 필요하지만, Stanford Cars는 24~68 정도로 균형이 괜찮아서 그대로 쓸 수 있다.

---

### Cell 4 — 샘플 이미지 시각화

```
┌─────┬─────┬─────┬─────┐
│ Car │ Car │ Car │ Car │
├─────┼─────┼─────┼─────┤
│ Car │ Car │ Car │ Car │   ← 12개 자동차 그리드
├─────┼─────┼─────┼─────┤
│ Car │ Car │ Car │ Car │
└─────┴─────┴─────┴─────┘
```

```python
from torchvision import transforms

try:
    _bilinear = Image.Resampling.BILINEAR
except AttributeError:
    _bilinear = Image.BILINEAR

vis_tf = transforms.Compose([
    transforms.Resize((224, 224), _bilinear),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
vis_trainset = StanfordCarsDataset(root=str(DATA_ROOT), split="train", transform=vis_tf)

show_sample_grid(vis_trainset, class_names=class_names, n=12, cols=4)
```

**실제 출력:** 12개 자동차 이미지가 4x3 그리드로 표시됨 (각 이미지 위에 클래스 이름 라벨)

**코드 설명:**
- `try/except AttributeError`: 새 Pillow 버전(`Image.Resampling.BILINEAR`)과 옛 버전(`Image.BILINEAR`) 둘 다 지원
- `transforms.Resize((224, 224), _bilinear)`: 모든 이미지를 224x224로 통일 (시각화용 작은 사이즈)
- `transforms.ToTensor()`: PIL 이미지 → PyTorch Tensor (0~1 범위로 자동 정규화)
- `transforms.Normalize([0.485,...], [0.229,...])`: ImageNet 평균/표준편차로 정규화 — 사전학습 모델의 입력 분포에 맞추기 위함
- `show_sample_grid(... n=12, cols=4)`: 12장을 4열로 그려서 한눈에 보여줌

> **왜 시각화하는가?** "내가 옳은 데이터셋을 로드했나?"를 5초 만에 확인. 만약 사진 대신 깨진 이미지나 엉뚱한 라벨이 보이면 데이터 로딩 코드에 버그가 있는 것.

---

### Cell 5 — 하이퍼파라미터 설정

```
┌────────────────────────────────────────┐
│  CFG = { ... 모든 설정값을 한곳에 ... } │
└────────────────────────────────────────┘
```

```python
CFG = {
    "model_type"  : "ViT-B_16",
    "split"       : "overlap",
    "slide_step"  : 12,
    "img_size"    : 448,
    "num_classes" : 196,          # Stanford Cars: 196종
    "train_batch" : 8,
    "eval_batch"  : 8,
    "num_steps"   : 10000,
    "quick_steps" : 200,
    "lr"          : 3e-2,
    "warmup_steps": 500,
    "decay_type"  : "cosine",
    "smoothing"   : 0.0,
    "fp16"        : True,
    "run_name"    : "transfg_cars",
    "num_workers" : 4,
}

print("=== TransFG Stanford Cars Config ===")
for k, v in CFG.items():
    print(f"  {k:20s}: {v}")
```

**실제 출력:** 위 딕셔너리의 모든 키-값이 정렬되어 출력됨

**코드 설명 (핵심 위주):**
- `"model_type": "ViT-B_16"`: ViT-Base 16x16 패치 모델 (Base = 86M 파라미터)
- `"split": "overlap"`: 패치를 겹쳐서 자름 (TransFG의 핵심 기법)
- `"slide_step": 12`: overlap 시 옆 패치와 12픽셀씩 슬라이드 (16-12=4픽셀 겹침)
- `"img_size": 448`: 입력 이미지 크기 — 일반 ImageNet(224)의 2배! 자동차 디테일을 보기 위함
- `"num_classes": 196`: **Stanford Cars 전용** (CUB는 200)
- `"train_batch": 8`: 작은 배치 크기 — 448x448 이미지가 GPU 메모리를 많이 먹어서
- `"num_steps": 10000`: 전체 학습 step 수 (논문 기준)
- `"quick_steps": 200`: 빠른 검증용 step 수 (~70분)
- `"lr": 3e-2`: 학습률 0.03 — Transformer 치고 큰 편이지만 SGD+momentum용
- `"warmup_steps": 500`: 처음 500 step은 lr을 0에서 천천히 올림 (학습 불안정 방지)
- `"decay_type": "cosine"`: warmup 후 cosine 곡선으로 lr 감소
- `"fp16": True`: Mixed precision — 16비트 부동소수점으로 메모리/속도 절약

> **왜 한곳에 모으는가?** 하이퍼파라미터를 코드 곳곳에 흩어 놓으면 "어디를 바꿨는지" 추적이 안 된다. CFG 딕셔너리에 모아두면 실험 재현·기록이 쉬움.

---

### Cell 6 — DataLoader 생성

```
Dataset (8,144장) ──→ DataLoader ──→ 배치 단위로 잘라서 모델에 공급
                       │
                       └─ shuffle, num_workers=4 (병렬 로딩)
```

```python
train_loader, test_loader, trainset, testset = get_cars_loaders(
    data_root        = str(DATA_ROOT),
    train_batch_size = CFG["train_batch"],
    eval_batch_size  = CFG["eval_batch"],
    img_size         = CFG["img_size"],
    num_workers      = CFG["num_workers"],
)

x_batch, y_batch = next(iter(train_loader))
print(f"Input batch  : {x_batch.shape}   dtype={x_batch.dtype}")
print(f"Label batch  : {y_batch.shape}   dtype={y_batch.dtype}")
print(f"Train 배치 수: {len(train_loader)} / epoch")
print(f"Test  배치 수: {len(test_loader)} / epoch")
```

**실제 출력:**
```
Input batch  : torch.Size([8, 3, 448, 448])   dtype=torch.float32
Label batch  : torch.Size([8])   dtype=torch.int64
Train 배치 수: 1018 / epoch
Test  배치 수: 1006 / epoch
```

**코드 설명:**
- `get_cars_loaders(...)`: 자동차 전용 DataLoader 생성 함수 (transform 자동 적용)
- 반환값 4개: train/test loader (배치 공급) + train/test set (Dataset 객체)
- `next(iter(train_loader))`: 첫 배치만 꺼내서 형태 확인
- `[8, 3, 448, 448]`: 배치 8장, RGB 3채널, 448x448 크기
- `dtype=torch.float32`: 32비트 실수 (정규화된 값)
- `[8]`, `int64`: 라벨 8개, 64비트 정수 (0~195 중 하나)
- `1018 = 8144 / 8`: 8,144장을 배치 8로 자르면 1018번 반복하면 1 epoch 완료

> **왜 num_workers=4인가?** 메인 프로세스가 GPU에 데이터 보내는 동안, 4개의 백그라운드 워커가 다음 배치를 미리 로드 → 학습 멈춤 없이 연속 처리.

---

### Cell 7 — 모델 생성

```
ViT-B_16 (사전학습) ──→ 분류 헤드 교체 (1000→196) ──→ TransFG 모델
       │
       └─ ImageNet 가중치 로드 → 무에서 학습보다 100배 빠르게 수렴
```

```python
config = CONFIGS[CFG["model_type"]]
config.split      = CFG["split"]
config.slide_step = CFG["slide_step"]

stride = config.slide_step
p = config.patches["size"][0]
n_patches_side = (CFG["img_size"] - p) // stride + 1
print(f"patch grid : {n_patches_side}×{n_patches_side} = {n_patches_side**2} patches")

model = VisionTransformer(
    config,
    img_size        = CFG["img_size"],
    zero_head       = True,
    num_classes     = CFG["num_classes"],   # 196
    smoothing_value = CFG["smoothing"],
)
model.load_from(np.load(str(PRETRAINED)))
model = model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 파라미터: {n_params/1e6:.1f}M")
print(f"분류 헤드    : {model.part_head}")

model.eval()
dummy = torch.randn(2, 3, CFG["img_size"], CFG["img_size"]).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f"추론 출력    : {out.shape}  (배치=2, 클래스=196)")
```

**실제 출력:**
```
patch grid : 37×37 = 1369 patches
학습 파라미터: 85.8M
분류 헤드    : Linear(in_features=768, out_features=196, bias=True)
추론 출력    : torch.Size([2, 196])  (배치=2, 클래스=196)
```

**코드 설명 (핵심 위주):**
- `CONFIGS[CFG["model_type"]]`: ViT-B_16 모델의 기본 설정 (레이어 수, hidden size 등)
- `config.split = "overlap"`, `config.slide_step = 12`: TransFG overlap 모드 활성화
- `n_patches_side = (448 - 16) // 12 + 1 = 37`: overlap split 공식 → 37x37=1369 패치
- `zero_head=True`: 분류 헤드(마지막 Linear) 가중치를 0으로 초기화 (1000클래스용 ImageNet 헤드 폐기)
- `num_classes=196`: 새 헤드는 196 출력 (Stanford Cars 전용)
- `model.load_from(np.load(PRETRAINED))`: ImageNet 사전학습 가중치 로드 (헤드 제외)
- `n_params/1e6 = 85.8M`: 8,580만 파라미터 (Base 모델)
- `dummy` 추론: 모델이 정상 작동하는지 가짜 데이터로 forward pass 테스트
- 출력 `[2, 196]`: 배치 2개에 대해 196개 클래스 점수

> **왜 사전학습 가중치를 쓰는가?** 1억 장의 ImageNet에서 이미 "이미지의 일반적 특징(엣지, 텍스처, 모양)"을 배운 모델을 가져와서, 마지막 분류기만 자동차용으로 새로 학습하면 훨씬 빠르고 정확하다. 처음부터 학습하면 자동차 8,144장으로는 부족하다.

---

### Cell 8 — Quick Training (200 steps)

```
[Step 1] ──→ [Step 50: 평가] ──→ [Step 100: 평가] ──→ [Step 200: 끝]
                  │                    │                    │
                  └─ Val Acc 기록       └─ Val Acc 기록       └─ Best 모델 저장
```

```python
import time

print(f"Quick training: {CFG['quick_steps']} steps on {DEVICE}")

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
print(f"\n경과 시간: {time.time()-start:.1f}초")
```

**실제 출력:**
```
Quick training: 200 steps on cuda
[Step   50/200] Train Loss: 5.3169 | Val Acc: 1.23% | Val Loss: 5.2693
[Step  100/200] Train Loss: 5.2855 | Val Acc: 1.14% | Val Loss: 5.2680
[Step  150/200] Train Loss: 5.2801 | Val Acc: 1.01% | Val Loss: 5.2630
[Step  200/200] Train Loss: 5.2917 | Val Acc: 1.37% | Val Loss: 5.2563
Best Val Acc: 1.37%
경과 시간: 4321.2초
```

**코드 설명 (핵심 위주):**
- `time.time()`: 시작 시간 기록 → 마지막에 차이를 구해서 경과 시간 출력
- `train(...)`: 학습 루프 실행 (forward, loss, backward, optimizer step 반복)
- `num_steps=200`: 200번만 forward/backward (200 batches)
- `warmup_steps=50`: 처음 50 step은 lr을 천천히 올림 (200 step 학습이라 50으로 줄임)
- `eval_every=50`: 50 step마다 test set으로 평가 → history에 기록
- `fp16=True`: mixed precision 학습 (메모리/속도 이득)
- `max_grad_norm=1.0`: gradient clipping — gradient가 폭주(exploding)하지 않도록 노름 1.0으로 제한
- `best_acc, history`: 최고 정확도와 step별 loss/acc 기록

**결과 해석:**
- 1.37% 정확도 = 랜덤(1/196 ≈ 0.51%)보다 약간 높음 → 학습은 시작됨
- 하지만 한참 미수렴 — **10,000 step은 돌려야 96.1%에 도달**
- Loss가 5.25에서 천천히 내려가는 중

> **왜 Quick Training을 먼저 하는가?** 코드 버그가 있으면 10시간 학습 후 발견하면 끔찍하다. 200 step (~70분)으로 "loss가 떨어지긴 하나?"만 빠르게 검증한 뒤 안전하면 전체 학습 진행.

---

### Cell 9 — 학습 History 시각화

```
Loss
 │  \
 │   \___
 │       \____
 │            \____
 └────────────────── step
```

```python
plot_history(history, eval_every=50)
```

**실제 출력:** 두 개의 그래프 (Train/Val Loss 곡선, Val Accuracy 곡선)

**코드 설명:**
- `plot_history(history, eval_every=50)`: history 딕셔너리에서 loss/acc를 꺼내 matplotlib으로 그래프 그림
- `eval_every=50`: 50 step마다 평가했으므로 x축이 [50, 100, 150, 200]

> **왜 그래프로 보는가?** 숫자만 보면 "5.3169 → 5.2917"이 줄어드는 건지 헷갈린다. 그래프로 보면 추세가 한눈에 보인다.

---

### Cell 10 — 전체 학습 (주석 처리, Optional)

```python
# 전체 학습 — 주석 해제 후 실행 (~8~12시간)
# model.load_from(np.load(str(PRETRAINED)))
# best_acc, history = train(num_steps=CFG["num_steps"], ...)
print("전체 학습: 위 주석 해제 후 실행")
print("TensorBoard: tensorboard --logdir logs")
```

**코드 설명:**
- 의도적으로 주석 처리 — 노트북 실행 시 자동으로 8~12시간 학습되면 곤란하니까
- `model.load_from(...)`: Quick training으로 망가진 가중치를 다시 사전학습으로 리셋해야 함
- `tensorboard --logdir logs`: 별도 터미널에서 실행 → 브라우저로 학습 곡선 모니터링

> **왜 reload하는가?** Quick training 200 step에서 lr이 너무 높아 가중치가 약간 망가졌을 수 있음. 전체 학습 전에 깨끗한 사전학습 가중치로 리셋하는 게 안전.

---

### Cell 11 — 체크포인트 로드

```
파일 있음?  ── Yes ──→ load_state_dict ──→ eval 모드
       │
       └── No ───→ 안내 메시지만 출력
```

```python
ckpt_path = OUTPUT_DIR / f"{CFG['run_name']}_checkpoint.bin"

if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"체크포인트 로드: {ckpt_path}")
else:
    print(f"체크포인트 없음: {ckpt_path}")
    model.eval()
```

**실제 출력:**
```
체크포인트 없음: output/stanford_cars/transfg_cars_checkpoint.bin
```

**코드 설명:**
- `OUTPUT_DIR / f"..."`: pathlib의 `/` 연산자로 경로 결합
- `ckpt_path.exists()`: 파일이 있으면 True
- `torch.load(..., map_location=DEVICE)`: 저장된 텐서를 현재 디바이스(GPU)로 직접 로드
- `model.load_state_dict(ckpt["model"])`: 저장된 가중치를 모델에 주입
- `model.eval()`: 평가 모드 (dropout/batchnorm 비활성)

> **왜 if/else로 처리하는가?** Quick training만 했을 경우 체크포인트가 없을 수 있음. 노트북이 에러로 멈추지 않고 진행되도록 graceful 처리.

---

### Cell 12 — 테스트셋 평가

```
모든 test 배치 ──→ forward ──→ argmax ──→ 정답과 비교 ──→ accuracy
```

```python
print("테스트셋 평가 중...")
result = evaluate_dataset(model, test_loader, DEVICE)

print(f"\n=== Stanford Cars 평가 결과 ===")
print(f"Test Accuracy : {result['accuracy']:.4f}  ({result['accuracy']*100:.2f}%)")
print(f"Test Avg Loss : {result['avg_loss']:.5f}")
print(f"\n※ TransFG 논문 최고 성능 (Stanford Cars)")
print(f"   overlap split, 10k steps: 96.1%")
```

**실제 출력:**
```
=== Stanford Cars 평가 결과 ===
Test Accuracy : 0.0137  (1.37%)
Test Avg Loss : 5.25628
※ TransFG 논문 최고 성능: 96.1%
```

**코드 설명:**
- `evaluate_dataset(model, test_loader, DEVICE)`: 모든 test 배치를 한 번씩 보고 정확도/loss 평균 계산
- `result['accuracy']`: 0~1 사이 비율 → `*100`해서 퍼센트로 출력
- 1.37%는 200 step 결과라 매우 낮음 — 정상

> **왜 96.1%와 비교하는가?** "이게 좋은 결과인가?"를 판단하려면 기준점이 필요. 논문 SOTA를 적어두면 추후 전체 학습 후 비교가 명확해진다.

---

### Cell 13 — 단일 이미지 추론

```
random idx ──→ PIL 이미지 ──→ transform ──→ model ──→ softmax
                                                       │
                                                       └─→ Top-5 클래스 + 확률
```

```python
idx = random.randint(0, len(testset_raw) - 1)
pil_img, true_label = testset_raw[idx]

from torchvision import transforms
infer_tf = transforms.Compose([
    transforms.Resize((600, 600), _bilinear),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

x = infer_tf(pil_img).unsqueeze(0).to(DEVICE)
model.eval()
with torch.no_grad():
    logits = model(x)
probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
pred  = int(probs.argmax())
top5_idx = probs.argsort()[::-1][:5]

print(f"실제 클래스 : [{true_label:3d}] {class_names[true_label]}")
print(f"예측 클래스 : [{pred:3d}] {class_names[pred]}")
print(f"신뢰도      : {probs[pred]:.4f}")
print(f"정답 여부   : {'✓ 정답' if pred == true_label else '✗ 오답'}")
print("\nTop-5 예측:")
for i in top5_idx:
    bar = '█' * int(probs[i] * 40)
    print(f"  {probs[i]:.4f} {bar} {class_names[i]}")
```

**실제 출력 (예):** 자동차 이미지가 표시되고, 아래에 Top-5 예측이 막대그래프 텍스트로 출력됨. 미수렴 모델이라 대부분 오답.

**코드 설명 (핵심 위주):**
- `random.randint(0, ...)`: test set에서 랜덤 인덱스 선택
- `Resize(600,600)` → `CenterCrop(448,448)`: 이미지를 약간 크게 만든 뒤 중앙 부분만 자름 (학습 시 transform과 일치)
- `unsqueeze(0)`: (3,448,448) → (1,3,448,448) — 배치 차원 추가
- `torch.no_grad()`: gradient 계산 끔 (추론은 학습 아님 → 메모리/속도 이득)
- `torch.softmax(logits, dim=-1)`: 점수를 확률로 변환 (합=1)
- `probs.argmax()`: 가장 높은 확률의 인덱스 = 예측 클래스
- `probs.argsort()[::-1][:5]`: 확률 내림차순 정렬 후 상위 5개
- `'█' * int(probs[i] * 40)`: 확률을 막대그래프로 시각화 (40칸 만점)

> **왜 Top-5를 보는가?** 모델이 1등을 틀렸어도 2~5등에 정답이 있으면 "헷갈리긴 하지만 가까이는 갔다"는 의미. Top-5도 다 틀리면 모델이 완전 헛다리 짚는 중.

---

### Cell 14 — 배치 예측 시각화

```
[Car 1: ✓]  [Car 2: ✓]  [Car 3: ✗]  [Car 4: ✓]   ← 초록 테두리 / 빨강 테두리
[Car 5: ✓]  [Car 6: ✗]  [Car 7: ✓]  [Car 8: ✓]
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

**실제 출력:** 8장의 자동차 이미지 그리드, 각 이미지 위에 [예측 / 정답] 라벨 표시 (정답이면 초록, 오답이면 빨강 테두리)

**코드 설명:**
- `next(iter(test_loader))`: test loader에서 첫 배치만 꺼냄
- `predict_batch(model, x_vis, DEVICE)`: 한 번에 8장을 모델에 통과시켜 예측
- `(preds == y_vis.numpy()).mean()`: 배치 안의 정확도 계산
- `x_vis[:8]`: 처음 8장만 시각화
- `visualize_predictions(...)`: 정답/오답을 색깔로 구분해 표시

> **왜 배치 단위로 보는가?** 단일 이미지 추론보다 빠르고, 모델이 어떤 종류의 자동차를 잘 맞추고 어떤 걸 헷갈려하는지 패턴이 보임.

---

### Cell 15 — Attention Map 시각화

```
입력 이미지       Layer 1 attention   Layer 6 attention   Layer 12 attention
  ┌─────┐          ┌─────┐             ┌─────┐             ┌─────┐
  │ Car │   ──→    │ ░▒▓ │   ──→       │ ░▒▒ │   ──→       │ ░ ▓ │
  │     │          │ ▒▒░ │             │ ▒▓░ │             │ ▓▒░ │  ← 그릴/휠에 집중
  └─────┘          └─────┘             └─────┘             └─────┘
                  (전체적)            (부위별)             (식별 부위)
```

```python
from data_loader_cars import get_cars_transforms
_, test_tf = get_cars_transforms(CFG["img_size"])
attn_testset = StanfordCarsDataset(str(DATA_ROOT), split="test", transform=test_tf)

idx = 0
img_tensor, label = attn_testset[idx]
print(f"클래스: [{label}] {class_names[label]}")
print("레이어별 CLS→Patch attention map (head=0)")

visualize_attention(
    model, img_tensor, DEVICE,
    head=0, patch_size=16, img_size=CFG["img_size"]
)
```

**실제 출력:** 12개 Transformer 레이어 각각의 attention 히트맵 — 깊은 레이어일수록 자동차의 그릴, 헤드라이트, 휠 등 식별 부위에 집중됨

**코드 설명:**
- `get_cars_transforms(...)`: Stanford Cars용 train/test transform 생성
- `_, test_tf`: train_tf는 무시, test_tf만 사용
- `attn_testset[0]`: 첫 번째 test 이미지를 transform 적용해 (Tensor, label) 반환
- `visualize_attention(model, img_tensor, ...)`: 모델 내부의 각 레이어 attention을 추출해 히트맵 오버레이
- `head=0`: 12개 attention head 중 0번만 시각화
- `patch_size=16`: 패치 크기 (히트맵을 원본 크기로 업샘플링할 때 필요)

> **왜 attention을 보는가?** 모델이 "어디를 보고 판단했는지" 시각적으로 확인. 만약 자동차 옆 사람이나 배경에 attention이 쏠리면 모델이 잘못 학습된 것. 식별 부위(그릴, 휠)에 집중하면 잘 학습된 것.

---

### Cell 16 — 12개 Head Attention 비교

```
Head 0    Head 1    Head 2    Head 3
┌────┐    ┌────┐    ┌────┐    ┌────┐
│그릴│    │휠  │    │전체│    │창문│   ← 각 head가 다른 부위에 집중!
└────┘    └────┘    └────┘    └────┘

Head 4    ...                  Head 11
```

```python
from visualization import AttentionMapExtractor

idx = 5
img_tensor, label = attn_testset[idx]
x = img_tensor.unsqueeze(0).to(DEVICE)

model.eval()
with AttentionMapExtractor(model) as ext:
    with torch.no_grad():
        _ = model(x)
    attn_maps = ext.attention_maps

if attn_maps:
    last = attn_maps[-1][0]          # (12, seq, seq)
    cls_attn = last[:, 0, 1:].numpy()  # (12, n_patches)
    n_p = int(cls_attn.shape[1] ** 0.5)
    if n_p * n_p == cls_attn.shape[1]:
        from visualization import denormalize
        from PIL import Image as PILImage
        img_rgb = denormalize(img_tensor)
        n_heads = last.shape[0]
        cols, rows = 4, (n_heads + 3) // 4
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
        plt.suptitle(f"12 Heads — {class_names[label]}", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print(f"overlap split: patch수({cls_attn.shape[1]})가 정방형 아님")
```

**실제 출력:** 4x3 그리드로 12개 head의 attention 히트맵이 표시됨. 각 head가 자동차의 다른 부위에 집중하는 패턴이 보임

**코드 설명 (핵심 위주):**
- `AttentionMapExtractor(model)`: with 블록 안에서 모델의 attention을 자동 캡처하는 컨텍스트 매니저
- `_ = model(x)`: 출력은 버리고 attention만 캡처
- `attn_maps[-1]`: 마지막(12번째) Transformer 레이어의 attention. 모양 (1, 12, seq, seq)
- `last[:, 0, 1:]`: 0번 토큰(CLS)이 1번부터 마지막 패치 토큰을 보는 attention만 추출
- `n_p = sqrt(n_patches)`: 1369=37x37이므로 n_p=37
- `(am - am.min()) / (am.max() - am.min())`: 0~1 정규화 (시각화 콘트라스트 향상)
- `PIL...resize(...)`: 37x37 attention map을 448x448로 업샘플링
- `imshow(img_rgb)` + `imshow(am_up, alpha=0.5, cmap="jet")`: 원본 위에 빨강~파랑 히트맵 오버레이
- `axes[h].axis("off")`: 축 숨김 (이미지 시각화는 축 불필요)

> **왜 12개 head를 따로 보는가?** Transformer의 multi-head attention은 "여러 시각으로 동시에 보는 메커니즘". head 0은 그릴에, head 1은 휠에, head 2는 전체 윤곽에 집중하는 식으로 분업한다. TransFG의 PSM은 이 head들의 정보를 종합해 "어떤 패치가 가장 식별력 있는가"를 결정한다.

---

### Cell 17 — CUB vs Stanford Cars 비교

```
┌──────────────┬────────────────┬─────────────────┐
│   항목        │ CUB-200-2011    │ Stanford Cars   │
├──────────────┼────────────────┼─────────────────┤
│ 도메인        │ 조류            │ 자동차           │
│ ...           │ ...             │ ...              │
└──────────────┴────────────────┴─────────────────┘
```

```python
print("=" * 55)
print(f"{'항목':20s} {'CUB-200-2011':>15s} {'Stanford Cars':>15s}")
print("=" * 55)
rows = [
    ("도메인",          "조류 (새)",       "자동차"),
    ("클래스 수",        "200종",           "196종"),
    ("Train",           "5,994장",         "8,144장"),
    ("Test",            "5,794장",         "8,041장"),
    ("논문 SOTA",        "91.7%",           "96.1%"),
    ("핵심 구별 부위",    "부리·깃털·눈",    "그릴·헤드라이트·휠"),
    ("배경 다양성",       "높음 (자연환경)",  "낮음 (주로 스튜디오)"),
]
for label, cub, cars in rows:
    print(f"  {label:18s} {cub:>15s} {cars:>15s}")
print("=" * 55)
print("\n→ Stanford Cars가 논문 기준 더 높은 정확도 달성")
print("  (배경이 단순해 모델이 차량 특징에 집중하기 유리)")
```

**실제 출력:** 두 데이터셋의 비교 표가 깔끔하게 정렬되어 출력됨

**코드 설명:**
- `"=" * 55`: 등호 55개로 구분선
- `f"{'항목':20s}"`: 문자열을 20칸으로 맞춤 (왼쪽 정렬)
- `f"{...:>15s}"`: 15칸 오른쪽 정렬
- `rows`: 비교할 항목들을 튜플 리스트로 정리
- for 루프로 줄마다 정렬된 표 형식 출력

> **왜 비교 표를 만드는가?** 같은 모델(TransFG)이 데이터셋에 따라 어떻게 다른 결과를 내는지 직관적으로 비교. 향후 다른 FGVC 데이터셋(Aircraft, Dogs 등)을 추가할 때도 같은 형식으로 확장 가능.

---

## 7. 공부 팁

### 7.1 단계별 학습 순서

1. **먼저 일반 ViT 이해** → 16x16 패치 + Transformer 인코더 + CLS 토큰
2. **그 다음 TransFG 차이점** → overlap split + PSM (Part Selection Module)
3. **마지막에 코드 디테일** → DataLoader, transform, lr scheduling

### 7.2 노트북 실행 순서 권장

```
Cell 1~2 (환경/임포트)            [필수, 30초]
   │
Cell 3~4 (데이터 확인)             [필수, 1분]
   │
Cell 5~7 (설정/모델 생성)          [필수, 2분]
   │
Cell 8 (Quick training, ~70분)    [건너뛰어도 OK]
   │
Cell 11 (체크포인트 로드)          [있으면 실행]
   │
Cell 12~16 (평가/시각화)           [필수, 5분]
   │
Cell 17 (비교 표)                 [선택]
```

### 7.3 자주 막히는 부분 (Stanford Cars 특화)

| 증상 | 원인 | 해결 |
|---|---|---|
| `assert DATA_ROOT.exists()` 실패 | data/Stanford_Cars 폴더 없음 | parquet 파일 다운로드 확인 |
| `num_classes` 불일치 에러 | 200으로 설정함 (CUB 값) | **196으로 수정** |
| OOM (메모리 부족) | batch_size 너무 큼 | `train_batch=4`로 감소 |
| Quick training Val Acc=0% | warmup_steps 너무 김 | 200 step 학습은 warmup 50으로 |
| Attention map이 정방형 아님 | overlap split 사용 중 | `n_p*n_p != n_patches` 체크 |

### 7.4 다음 단계 도전 과제

- [ ] **전체 학습** 돌려서 96.1%에 얼마나 가까이 가는지 확인
- [ ] **다른 모델 비교**: ViT-B_16 → ViT-L_16 (Large 모델, 304M 파라미터)
- [ ] **데이터 증강 추가**: RandAugment, Mixup, CutMix
- [ ] **TTA (Test Time Augmentation)**: 추론 시 5번 다른 crop으로 평가 후 평균
- [ ] **Confusion Matrix**: 어떤 자동차들이 서로 헷갈리는지 분석 (예: BMW 3시리즈 2010 ↔ 2012)

### 7.5 핵심 한 문장

> **TransFG는 ViT에 "어떤 패치가 중요한지 고르는 능력"을 더한 모델이고, Stanford Cars처럼 배경이 단순하고 식별 부위가 명확한 데이터셋에서 특히 잘 작동한다.**

---

작성: AiffelThon01 프로젝트  
참고 노트북: `TransFG_Stanford_Cars.ipynb`  
관련 노트북: `TransFG_CUB.ipynb`, `TransFG_FGVC_Aircraft.ipynb`
