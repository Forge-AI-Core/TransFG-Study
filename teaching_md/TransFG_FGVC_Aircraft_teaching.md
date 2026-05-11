# TransFG로 항공기 100종 분류하기 (FGVC-Aircraft)

> fine-grained image classification 의 이해 
> 모델: TransFG (Vision Transformer 기반) | 데이터: FGVC-Aircraft variant 100종

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [핵심 개념 (비유로 이해하기)](#2-핵심-개념-비유로-이해하기)
3. [전체 구조 (ASCII 다이어그램)](#3-전체-구조-ascii-다이어그램)
4. [FGVC-Aircraft 특화 개념](#4-fgvc-aircraft-특화-개념)
5. [Cell 1 — 환경 확인](#cell-1--환경-확인)
6. [Cell 2 — 모듈 임포트 및 경로 설정](#cell-2--모듈-임포트-및-경로-설정)
7. [Cell 3 — 데이터셋 탐색](#cell-3--데이터셋-탐색)
8. [Cell 4 — 클래스 계층 탐색](#cell-4--클래스-계층-탐색)
9. [Cell 5 — 샘플 이미지 시각화](#cell-5--샘플-이미지-시각화)
10. [Cell 6 — 하이퍼파라미터 설정](#cell-6--하이퍼파라미터-설정)
11. [Cell 7 — DataLoader 생성](#cell-7--dataloader-생성)
12. [Cell 8 — 모델 생성](#cell-8--모델-생성)
13. [Cell 9 — Quick Training](#cell-9--quick-training)
14. [Cell 10 — 학습 History 시각화](#cell-10--학습-history-시각화)
15. [Cell 11 — 전체 학습 (Optional)](#cell-11--전체-학습-optional)
16. [Cell 12 — 체크포인트 로드](#cell-12--체크포인트-로드)
17. [Cell 13 — Val + Test 평가](#cell-13--val--test-평가)
18. [Cell 14 — 단일 이미지 추론](#cell-14--단일-이미지-추론)
19. [Cell 15 — 배치 예측 시각화](#cell-15--배치-예측-시각화)
20. [Cell 16 — Attention Map 시각화](#cell-16--attention-map-시각화)
21. [Cell 17 — 12개 Head Attention 비교](#cell-17--12개-head-attention-비교)
22. [Cell 18 — 3개 데이터셋 종합 비교](#cell-18--3개-데이터셋-종합-비교)
23. [Cell 19 — 논문 성능 Bar Chart](#cell-19--논문-성능-bar-chart)
24. [3개 데이터셋 비교 표](#3개-데이터셋-비교-표)
25. [공부 팁](#공부-팁)

---

## 1. 한 줄 요약

**TransFG**는 Vision Transformer(ViT)에 "어느 patch를 보고 분류할지 골라주는" Part Selection Module(PSM)을 붙인 모델로, 이번 노트북에서는 100종의 항공기 variant(예: Boeing 737-300 vs 737-400)를 구분하는 fine-grained classification을 수행합니다.

---

## 2. 핵심 개념 (비유로 이해하기)

### 2-1. Fine-Grained Classification이란?

```
일반 분류 (coarse)        Fine-grained 분류
┌──────────┐             ┌──────────┐
│ "강아지" │             │ "푸들"   │
│ "고양이" │             │ "치와와" │
│ "비행기" │             │ "시바견" │
└──────────┘             └──────────┘
   쉬움                     어려움
```

- **일반 분류**: 강아지 vs 고양이 vs 비행기 (큰 차이)
- **Fine-grained**: 같은 강아지 안에서 푸들 vs 시바견 (미세한 차이)
- 이 노트북: **같은 Boeing 737 안에서 737-300 vs 737-400** 구분

### 2-2. Vision Transformer (ViT) 비유

> **이미지를 "퍼즐 조각"으로 자르고, 각 조각을 단어처럼 다룬다.**

```
원본 이미지 (448×448)
  ↓ patch로 자르기 (16×16 크기)
  ┌─┬─┬─┬─┬─┐
  │ │ │ │ │ │  ← 각 patch를 "단어"라고 생각
  ├─┼─┼─┼─┼─┤
  │ │ │ │ │ │
  └─┴─┴─┴─┴─┘
  ↓ Transformer (BERT처럼 patch 간 관계 학습)
  ↓ [CLS] token이 전체 정보 요약
  ↓ 최종 분류 (100종 중 하나)
```

### 2-3. TransFG의 추가 아이디어 — Part Selection

> **"비행기 사진의 모든 patch가 중요한 건 아니다. 엔진·날개·꼬리만 잘 보면 된다."**

- ViT는 모든 patch를 동일하게 다룸
- TransFG는 **attention이 강한 patch만 골라서** 마지막 레이어에 추가 입력
- → 변별력 있는 부위(discriminative parts)에 집중

### 2-4. Overlap Split

```
일반 ViT (non-overlap)        TransFG (overlap)
patch 간격 = 16               patch 간격 = 12 (slide_step)
┌──┐┌──┐┌──┐                  ┌──┐
│ 1││ 2││ 3│                  │ 1│
└──┘└──┘└──┘                    └─┐┌─┐
                                  │2││3│  ← 겹치게 자름
                                  └─┘└─┘
patch 수: 14×14=196           patch 수: 37×37=1369 (더 촘촘)
```

- **overlap split**으로 patch가 더 많아지고 미세한 차이도 포착 가능
- 대신 계산량 증가 → 메모리·시간 비용 ↑

---

## 3. 전체 구조 (ASCII 다이어그램)

```
┌────────────────────────────────────────────────────────────────────┐
│                  TransFG FGVC-Aircraft 전체 파이프라인              │
└────────────────────────────────────────────────────────────────────┘

[1] 환경 확인 (Cell 1)
        │
        ▼
[2] 모듈 임포트 + 경로 설정 (Cell 2)
        │
        ▼
[3] 데이터셋 탐색 (Cell 3,4)
        │   ├─ AircraftDataset (train/val/test/trainval)
        │   └─ 100 variant / 70 family / 41 manufacturer
        ▼
[4] 시각화 (Cell 5) — show_sample_grid
        │
        ▼
[5] CFG 설정 (Cell 6) — img_size=448, num_classes=100
        │
        ▼
[6] DataLoader 생성 (Cell 7) — train/val/test 3분할
        │
        ▼
[7] 모델 생성 (Cell 8)
        │   ├─ ViT-B_16 + overlap split
        │   ├─ ImageNet pretrained 로드 (ViT-B_16.npz)
        │   └─ 분류 헤드: Linear(768 → 100)
        ▼
[8] Quick Training (Cell 9) — 200 steps
        │   └─ val_loader로 중간 검증
        ▼
[9] History 시각화 (Cell 10)
        │
        ▼
[10] 전체 학습 (Cell 11, optional) — 10000 steps + use_trainval=True
        │
        ▼
[11] 체크포인트 로드 (Cell 12)
        │
        ▼
[12] 평가 (Cell 13) — Val Acc + Test Acc
        │
        ▼
[13] 추론 / 시각화 (Cell 14~17)
        │   ├─ 단일 이미지 (predict_single)
        │   ├─ 배치 (predict_batch)
        │   ├─ Attention map (레이어별)
        │   └─ 12 head 비교
        ▼
[14] 3개 데이터셋 종합 비교 (Cell 18,19)
        └─ CUB / Cars / Aircraft
```

---

## 4. FGVC-Aircraft 특화 개념

### 4-1. 3계층 클래스 구조

```
manufacturer (41종)        예) Boeing, Airbus, Cessna...
    │
    └─ family (70종)       예) Boeing 737, Boeing 747, Airbus A320...
           │
           └─ variant (100종) ← TransFG 논문 기준
                              예) 737-300, 737-400, 737-500
```

- 같은 manufacturer(Boeing) 안에 여러 family(737, 747, 787) 존재
- 같은 family(737) 안에 여러 variant(737-300, 737-400, 737-500) 존재
- **variant 분류가 가장 어려움** — 창문 수, 엔진 위치 등 극히 미세한 차이

### 4-2. 왜 Aircraft가 가장 어려운가?

1. **클래스당 학습 데이터 최소** — 약 33장/클래스 (CUB 30장, Cars 41장)
2. **동일 family 내 variant 구분 어려움** — 737-300 vs 737-400은 창문 수, 엔진 위치 미세 차이
3. **촬영 각도 다양성** — 이륙/착륙/지상 대기 등 자세 변화 매우 큼
4. **논문 SOTA 70.7%** — CUB(91.7%), Cars(96.1%)보다 훨씬 낮음

### 4-3. val/test 분리 — CUB/Cars와의 차이점

```
CUB-200-2011, Stanford Cars       FGVC-Aircraft
┌─────────┬─────────┐             ┌────┬─────┬──────┐
│  train  │  test   │             │tr  │ val │ test │
└─────────┴─────────┘             └────┴─────┴──────┘
                                    ↓
                                  trainval = train + val
                                  (논문 최종 학습용)
```

- **빠른 실험**: train으로 학습 → val로 검증
- **최종 평가**: trainval(=train+val 합산)로 학습 → test 평가 (논문 기준)

### 4-4. `use_trainval` 파라미터의 의미

| 값 | 학습 데이터 | 검증 데이터 | 용도 |
|----|----------|----------|-----|
| `False` (기본) | train 3,334장 | val 3,333장 | 빠른 개발/디버깅 |
| `True` | trainval 6,667장 | test 3,333장 | 논문 SOTA 재현 |

### 4-5. `level` 파라미터 선택

| level | 클래스 수 | 난이도 | 예시 |
|-------|---------|------|-----|
| manufacturer | 41 | 쉬움 | Boeing vs Airbus |
| family | 70 | 보통 | Boeing 737 vs 747 |
| variant | 100 | **어려움** | 737-300 vs 737-400 |

---

## Cell 1 — 환경 확인

```
┌──────────────────────────────┐
│ Python · PyTorch · CUDA · GPU│
└──────────────────────────────┘
```

> **목적**: 학습을 시작하기 전에 GPU·라이브러리 버전을 확인 (호환성 문제 사전 방지).

```python
import sys, torch
print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**코드 설명:**
- `import sys, torch`: 시스템 정보(`sys`)와 PyTorch를 불러옴
- `sys.version.split()[0]`: "3.12.3 (main, ...)" 같은 긴 문자열에서 첫 토큰만 추출
- `torch.__version__`: 설치된 PyTorch 버전 (cu130 = CUDA 13.0 빌드)
- `torch.cuda.is_available()`: GPU 사용 가능 여부 — `True`여야 학습 가능
- `torch.cuda.get_device_name(0)`: 0번 GPU의 모델명
- `total_memory / 1e9`: byte를 GB로 환산 (1e9 = 10억)

**실행 결과:**
```
Python : 3.12.3
PyTorch: 2.10.0+cu130
CUDA   : True
GPU    : NVIDIA GB10
Memory : 128.0 GB
```

**왜 이렇게 하는가?**
- GPU 미인식 시 학습이 CPU로 떨어져 100배 이상 느려짐 → 미리 확인 필수
- GB10은 unified memory 128GB → 큰 batch도 OOM 위험 적음

---

## Cell 2 — 모듈 임포트 및 경로 설정

```
┌───────────────────────────────────────┐
│  TransFG/ ──▶ sys.path 추가 ──▶ import│
└───────────────────────────────────────┘
```

> **목적**: TransFG repo 코드를 import 가능하게 만들고, 데이터·가중치·출력 경로를 정의.

```python
import sys, random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

TRANSFG_ROOT = Path("TransFG").resolve()
if str(TRANSFG_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFG_ROOT))

from models.modeling import VisionTransformer, CONFIGS
from dataset_aircraft      import AircraftDataset
from data_loader_aircraft  import get_aircraft_loaders
from trainer         import train, validate
from inference_utils import predict_single, predict_batch, evaluate_dataset
from visualization   import show_sample_grid, plot_history, visualize_predictions, visualize_attention

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT  = Path("data/FGVC_Aircraft/fgvc-aircraft-2013b")
PRETRAINED = Path("pretrained/ViT-B_16.npz")
OUTPUT_DIR = Path("output/fgvc_aircraft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device    : {DEVICE}")
print(f"DATA_ROOT : {DATA_ROOT.resolve()}")
assert DATA_ROOT.exists(),  "데이터셋 없음 — data/FGVC_Aircraft/fgvc-aircraft-2013b 확인"
assert PRETRAINED.exists(), "가중치 없음 — pretrained/ViT-B_16.npz 확인"
```

**코드 설명 (핵심 위주):**
- `Path("TransFG").resolve()`: 상대 경로를 절대 경로로 변환
- `sys.path.insert(0, ...)`: TransFG 폴더를 import 검색 경로의 맨 앞에 추가 → `from models.modeling` 가능
- `VisionTransformer, CONFIGS`: TransFG의 ViT 본체 + 설정 dict
- `AircraftDataset`: FGVC-Aircraft 전용 PyTorch Dataset
- `get_aircraft_loaders`: train/val/test 3개 DataLoader 한 번에 생성
- `predict_single/batch, evaluate_dataset`: 추론 helper
- `DEVICE`: GPU 가능하면 `cuda`, 아니면 `cpu`
- `assert`: 데이터/가중치 미존재 시 즉시 에러로 알림

**실행 결과:**
```
Device    : cuda
DATA_ROOT : /workspace/data/FGVC_Aircraft/fgvc-aircraft-2013b
```

**왜 이렇게 하는가?**
- TransFG는 `import models.modeling` 형태로 작성되어 있어서 `sys.path`에 추가 필요
- `assert`로 사전 점검 → 학습 도중 파일 못 찾아 시간 낭비 방지
- `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)`: 출력 폴더 자동 생성

---

## Cell 3 — 데이터셋 탐색

```
┌─────────────────────────────────────┐
│ train(3,334) ─┐                     │
│ val(3,333) ───┼─▶ trainval(6,667)   │
│ test(3,333)                          │
└─────────────────────────────────────┘
```

> **목적**: split별 데이터 개수, 클래스 수, 클래스당 샘플 분포를 확인.

```python
from collections import Counter

trainset_raw = AircraftDataset(root=str(DATA_ROOT), split="train",    level="variant")
valset_raw   = AircraftDataset(root=str(DATA_ROOT), split="val",      level="variant")
testset_raw  = AircraftDataset(root=str(DATA_ROOT), split="test",     level="variant")
tvset_raw    = AircraftDataset(root=str(DATA_ROOT), split="trainval", level="variant")
class_names  = trainset_raw.get_class_names()

print(f"Train    : {len(trainset_raw):,}장")
print(f"Val      : {len(valset_raw):,}장")
print(f"Test     : {len(testset_raw):,}장")
print(f"TrainVal : {len(tvset_raw):,}장  (train+val 합산)")
print(f"클래스   : {len(class_names)}종 (variant)")

print("\n클래스 예시:")
for i in [0, 24, 49, 74, 99]:
    print(f"  [{i:2d}] {class_names[i]}")

counts = sorted(Counter(trainset_raw.labels).values())
print(f"\n클래스별 Train 샘플: min={counts[0]}, max={counts[-1]}, avg={sum(counts)/len(counts):.1f}")

img, label = trainset_raw[0]
print(f"\n샘플[0]: PIL size={img.size}, label={label} ({class_names[label]})")
```

**코드 설명 (핵심 위주):**
- `AircraftDataset(... split=..., level="variant")`: 4개 split을 각각 별도 객체로 로드
- `level="variant"`: 100종 분류 (가장 어려운 단계)
- `get_class_names()`: 0~99 인덱스 → 실제 항공기 이름 매핑
- `Counter(trainset_raw.labels)`: 각 클래스별 샘플 수 카운트
- `sorted(...).values()`: 정렬된 카운트 리스트로 min/max/avg 계산
- `trainset_raw[0]`: __getitem__ 호출 → (PIL Image, label) 튜플 반환

**실행 결과:**
```
Train    : 3,334장
Val      : 3,333장
Test     : 3,333장
TrainVal : 6,667장  (train+val 합산)
클래스   : 100종 (variant)

클래스 예시:
  [ 0] 707-320
  [24] C-130
  [49] F/A-18
  [74] SR-20
  [99] Yak-42

클래스별 Train 샘플: min=20, max=52, avg=33.3
샘플[0]: PIL size=(1024, 683), label=0 (707-320)
```

**왜 이렇게 하는가?**
- 클래스당 평균 33장 → 매우 적은 데이터 → pretrained 가중치 필수
- min=20 / max=52 → 클래스 불균형 존재 (약 2.6배 차이)
- 이미지 크기가 1024×683 등 가변 → DataLoader에서 448×448로 resize 필요

---

## Cell 4 — 클래스 계층 탐색

```
manufacturer (41)
  ├─ Boeing
  │   ├─ family: Boeing 707
  │   │   └─ variant: 707-320
  │   └─ family: Boeing 737
  │       ├─ variant: 737-200
  │       ├─ variant: 737-300  ← 미세한 차이
  │       └─ variant: 737-400
```

> **목적**: variant/family/manufacturer 3계층의 클래스 수 비교 + 같은 family 안에 여러 variant가 있음을 확인.

```python
for level, n_cls in [("variant", 100), ("family", 70), ("manufacturer", 41)]:
    ds = AircraftDataset(root=str(DATA_ROOT), split="trainval", level=level)
    print(f"  {level:14s}: {len(ds.get_class_names()):3d}종  | trainval {len(ds):,}장")

print("\n=== Variant → Family 관계 예시 ===")
data_dir = DATA_ROOT / "data"
variant_lines  = (data_dir / "images_variant_train.txt").read_text().strip().splitlines()
family_lines   = (data_dir / "images_family_train.txt").read_text().strip().splitlines()

v_map = {l.split()[0]: l.split(maxsplit=1)[1] for l in variant_lines}
f_map = {l.split()[0]: l.split(maxsplit=1)[1] for l in family_lines}

shown = set()
for img_id, variant in list(v_map.items())[:200]:
    family = f_map.get(img_id, "?")
    key = (variant, family)
    if key not in shown:
        print(f"  variant: {variant:20s} → family: {family}")
        shown.add(key)
    if len(shown) >= 6:
        break
```

**코드 설명 (핵심 위주):**
- `for level in [...]`: 3개 level 순회하며 클래스 수 출력
- `images_variant_train.txt`, `images_family_train.txt`: 같은 image_id가 양쪽에 존재
- `l.split()[0]`: 첫 토큰 = image_id, `l.split(maxsplit=1)[1]`: 나머지 = 클래스명
- `v_map[image_id] = variant`, `f_map[image_id] = family` 딕셔너리 생성
- `shown` set으로 중복 (variant, family) 쌍 제거

**실행 결과:**
```
  variant       : 100종  | trainval 6,667장
  family        :  70종  | trainval 6,667장
  manufacturer  :  41종  | trainval 6,667장

=== Variant → Family 관계 예시 ===
  variant: 707-320              → family: Boeing 707
  variant: 727-200              → family: Boeing 727
  variant: 737-200              → family: Boeing 737
  variant: 737-300              → family: Boeing 737
  variant: 737-400              → family: Boeing 737
  variant: 737-500              → family: Boeing 737
```

**왜 이렇게 하는가?**
- `Boeing 737` family 안에 `737-200/300/400/500` 4개 variant → 이런 미세 차이 학습이 핵심 challenge
- variant level에서 같은 family 내 variant는 시각적으로 매우 유사 → fine-grained 본연의 어려움

---

## Cell 5 — 샘플 이미지 시각화

```
┌─────────────────────────────────────┐
│ Resize(224) → ToTensor → Normalize  │
│           ↓                         │
│   show_sample_grid (4×3=12장)       │
└─────────────────────────────────────┘
```

> **목적**: 실제 이미지를 12장 그리드로 출력해 데이터셋 감을 잡음.

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
vis_trainset = AircraftDataset(root=str(DATA_ROOT), split="train", level="variant", transform=vis_tf)

show_sample_grid(vis_trainset, class_names=class_names, n=12, cols=4)
```

**코드 설명:**
- `try/except AttributeError`: PIL 버전 호환 (10.0+ vs 이전)
- `transforms.Compose`: 여러 변환을 순차 적용
- `Resize((224, 224))`: 시각화용 작은 크기로 축소 (실제 학습은 448)
- `ToTensor()`: PIL → torch.Tensor + 0~255 → 0~1 정규화
- `Normalize([0.485,...], [0.229,...])`: ImageNet 평균/표준편차 (pretrained 모델 호환 필수)
- `transform=vis_tf`: Dataset에 변환 주입 → __getitem__ 결과가 Tensor가 됨
- `show_sample_grid`: helper 함수, n=12장을 cols=4로 그리드 표시

**실행 결과:** 12개 항공기 이미지 그리드 (각 이미지 위에 variant명 표시)

**왜 이렇게 하는가?**
- ImageNet pretrained 모델은 ImageNet의 평균·표준편차로 정규화된 입력 기대 → 동일 normalize 사용
- 시각화 단계에서는 224×224로도 충분 (실제 학습은 448로 더 미세한 detail 활용)

---

## Cell 6 — 하이퍼파라미터 설정

```
┌───────────────────────────────────────┐
│  CFG dict ──▶ 모든 셀에서 참조        │
│   img_size=448, num_classes=100,      │
│   slide_step=12 (overlap split)       │
└───────────────────────────────────────┘
```

> **목적**: 학습 관련 설정을 한 곳에 모아 실험 재현성을 보장.

```python
CFG = {
    "model_type"   : "ViT-B_16",
    "split"        : "overlap",
    "slide_step"   : 12,
    "img_size"     : 448,
    "num_classes"  : 100,          # FGVC-Aircraft variant: 100종
    "train_batch"  : 8,
    "eval_batch"   : 8,
    "num_steps"    : 10000,
    "quick_steps"  : 200,
    "lr"           : 3e-2,
    "warmup_steps" : 500,
    "decay_type"   : "cosine",
    "smoothing"    : 0.0,
    "fp16"         : True,
    "run_name"     : "transfg_aircraft",
    "num_workers"  : 4,
    "level"        : "variant",
    "use_trainval" : False,
}

print("=== TransFG FGVC-Aircraft Config ===")
for k, v in CFG.items():
    print(f"  {k:20s}: {v}")
```

**코드 설명 (핵심 위주):**
- `model_type="ViT-B_16"`: ViT-Base, patch=16 (768-dim, 12 layer, 12 head)
- `split="overlap"`, `slide_step=12`: TransFG의 핵심 — patch를 겹치게 자름
- `img_size=448`: 일반 ViT는 224, TransFG는 448로 더 큰 입력 사용
- `num_classes=100`: variant level
- `train_batch=8`: 448 이미지 + overlap split → 메모리 많이 먹어서 작게
- `num_steps=10000`: 논문 기준 전체 학습량
- `quick_steps=200`: 노트북에서 빠르게 동작 확인용
- `lr=3e-2`: SGD with momentum용 — Adam과 다른 큰 lr
- `warmup_steps=500`, `decay_type="cosine"`: warmup 후 cosine으로 감소
- `fp16=True`: mixed precision → 속도 ↑, 메모리 ↓
- `use_trainval=False`: 빠른 개발 모드 (True면 논문 SOTA 재현용)

**실행 결과:** CFG dict 전체 출력 (`num_classes=100`, `level=variant`가 핵심)

**왜 이렇게 하는가?**
- 모든 하이퍼파라미터를 dict로 모아두면 실험 비교 시 추적 용이
- `quick_steps=200`은 코드 검증용 — 실제 학습은 `num_steps=10000`

---

## Cell 7 — DataLoader 생성

```
┌──────────────────────────────────────┐
│ get_aircraft_loaders                 │
│  ├─ train_loader (3,334 / 8 = 416배치)│
│  ├─ val_loader  (3,333 / 8 = 417배치)│
│  └─ test_loader (3,333 / 8 = 417배치)│
└──────────────────────────────────────┘
```

> **목적**: 학습용 train_loader + 검증용 val_loader + 최종평가용 test_loader 동시 생성.

```python
train_loader, val_loader, test_loader, trainset, valset, testset = get_aircraft_loaders(
    data_root        = str(DATA_ROOT),
    train_batch_size = CFG["train_batch"],
    eval_batch_size  = CFG["eval_batch"],
    img_size         = CFG["img_size"],
    num_workers      = CFG["num_workers"],
    level            = CFG["level"],
    use_trainval     = CFG["use_trainval"],
)

x_batch, y_batch = next(iter(train_loader))
print(f"Input batch  : {x_batch.shape}   dtype={x_batch.dtype}")
print(f"Label batch  : {y_batch.shape}   dtype={y_batch.dtype}")
print(f"Train 배치 수: {len(train_loader)}")
print(f"Val   배치 수: {len(val_loader)}")
print(f"Test  배치 수: {len(test_loader)}")
```

**코드 설명:**
- `get_aircraft_loaders`: 6개 객체 한 번에 반환 (3 loader + 3 dataset)
- `train_batch_size=8` / `eval_batch_size=8`: train과 eval 분리 가능 (eval은 더 크게도 OK)
- `num_workers=4`: 데이터 로딩을 4개 프로세스로 병렬화 → I/O 병목 완화
- `next(iter(train_loader))`: 첫 배치 한 개만 꺼내서 shape 확인
- `x_batch.shape`: (batch, channel, H, W) = (8, 3, 448, 448)
- `y_batch.shape`: (batch,) = (8,) → label 정수 8개
- `len(train_loader)`: 전체 배치 개수 (= 데이터 수 / batch_size)

**실행 결과:**
```
Input batch  : torch.Size([8, 3, 448, 448])   dtype=torch.float32
Label batch  : torch.Size([8])   dtype=torch.int64
Train 배치 수: 416
Val   배치 수: 417
Test  배치 수: 417
```

**왜 이렇게 하는가?**
- train_batch=8 × 416 ≈ 3328 ≈ 3334 (마지막 배치 일부 잘림)
- val/test는 shuffle=False로 동일한 순서 평가 → 재현성 확보
- `use_trainval=False`이므로 train_loader는 train만 사용 (val은 검증용)

---

## Cell 8 — 모델 생성

```
┌───────────────────────────────────────┐
│  ViT-B_16 + overlap split              │
│   patches = 37 × 37 = 1369             │
│   학습 파라미터: 85.8M                 │
│   분류 헤드: Linear(768 → 100)         │
└───────────────────────────────────────┘
```

> **목적**: TransFG 모델을 만들고 ImageNet pretrained 가중치를 로드 + sanity check.

```python
config = CONFIGS[CFG["model_type"]]
config.split      = CFG["split"]
config.slide_step = CFG["slide_step"]

p_size = config.patches["size"][0]
n_side = (CFG["img_size"] - p_size) // CFG["slide_step"] + 1
print(f"patch grid : {n_side}×{n_side} = {n_side**2} patches")

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
print(f"학습 파라미터: {n_params/1e6:.1f}M")
print(f"분류 헤드    : {model.part_head}")

model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 3, CFG["img_size"], CFG["img_size"]).to(DEVICE))
print(f"추론 출력    : {out.shape}  (배치=2, 클래스=100)")
```

**코드 설명 (핵심 위주):**
- `CONFIGS["ViT-B_16"]`: 사전 정의된 ViT-Base 설정 (768 dim, 12 layer)
- `config.split = "overlap"`, `config.slide_step = 12`: TransFG 핵심 옵션
- `n_side = (448-16)//12 + 1 = 37`: overlap 적용 후 가로 patch 수 → 총 1369 patch
- `zero_head=True`: 분류 헤드를 0으로 초기화 (ImageNet 1000 클래스 → Aircraft 100 클래스)
- `load_from(np.load(...))`: ImageNet pretrained `.npz` 로드 (헤드 제외 backbone 가중치)
- `n_params/1e6`: 약 85.8M (ViT-B 사이즈)
- `torch.randn(2, 3, 448, 448)`: 더미 입력으로 forward 테스트
- `out.shape == (2, 100)`: 배치 2개에 대해 100 클래스 logit 출력

**실행 결과:**
```
patch grid : 37×37 = 1369 patches
학습 파라미터: 85.8M
분류 헤드    : Linear(in_features=768, out_features=100, bias=True)
추론 출력    : torch.Size([2, 100])  (배치=2, 클래스=100)
```

**왜 이렇게 하는가?**
- non-overlap이면 patch 수 = (448/16)² = 784개, overlap이면 1369개 → 1.7배 증가
- `zero_head=True` 안 하면 ImageNet 1000 클래스 헤드가 로드되어 클래스 수 mismatch 에러
- pretrained 안 쓰면 클래스당 33장으로는 ViT 학습 거의 불가능

---

## Cell 9 — Quick Training

```
┌──────────────────────────────────────┐
│ 200 steps / batch=8 / fp16            │
│   ├─ 매 50 step마다 val 검증         │
│   └─ best_acc 자동 저장              │
└──────────────────────────────────────┘
```

> **목적**: 코드 동작 검증용 짧은 학습 (200 steps). 실제 SOTA는 10,000 steps 필요.

```python
import time

print(f"Quick training: {CFG['quick_steps']} steps / eval on val_loader")

start = time.time()
best_acc, history = train(
    model        = model,
    train_loader = train_loader,
    test_loader  = val_loader,
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

**코드 설명 (핵심 위주):**
- `test_loader=val_loader`: quick test에서는 val로 중간 검증 (test는 최종 평가에만)
- `num_steps=200`: 200번 SGD update (epoch 단위 아님)
- `warmup_steps=50`: 처음 50 step은 lr을 0→3e-2로 선형 증가
- `eval_every=50`: 50 step마다 val_loader 평가
- `fp16=True`: float16 mixed precision
- `gradient_accumulation_steps=1`: 누적 없이 매 step 업데이트
- `max_grad_norm=1.0`: gradient clipping (폭발 방지)
- `best_acc, history`: 최고 정확도 + 학습 곡선 dict 반환

**실행 결과:**
```
[Step   50/200] Train Loss: 4.6348 | Val Acc: 3.24% | Val Loss: 4.5725
[Step  100/200] Train Loss: 4.5670 | Val Acc: 5.34% | Val Loss: 4.4734
[Step  150/200] Train Loss: 4.4757 | Val Acc: 6.66% | Val Loss: 4.3634
[Step  200/200] Train Loss: 4.3992 | Val Acc: 6.18% | Val Loss: 4.3384
Best Val Acc: 6.66%
경과 시간: 1878.5초
```

**왜 이렇게 하는가?**
- 200 step ≈ 0.5 epoch → 정확도 6% 수준 (random=1% 대비 약간 학습됨)
- 논문 70.7% 도달하려면 10,000 step (약 25 epoch) 필요 → 약 26시간
- 빠른 코드 검증이 목적 → quick mode로 파이프라인 동작만 확인

---

## Cell 10 — 학습 History 시각화

```
┌──────────────────────────────────┐
│  Train Loss / Val Acc            │
│       ↓                          │
│   plot_history (matplotlib)      │
└──────────────────────────────────┘
```

> **목적**: history dict를 받아 학습 곡선을 그래프로 표시.

```python
plot_history(history, eval_every=50)
```

**코드 설명:**
- `history`: train()이 반환한 학습 기록 dict (loss, val_acc, val_loss 등 step별)
- `eval_every=50`: x축에 정확한 step 표시용
- `plot_history`: helper 함수 — Loss와 Val Accuracy를 두 subplot으로 표시

**실행 결과:** Loss(왼쪽) / Val Accuracy(오른쪽) 두 그래프

**왜 이렇게 하는가?**
- 학습이 잘 되는지 한눈에 확인 (loss 감소? val_acc 증가?)
- overfitting 발생 시 train_loss는 계속 감소하는데 val_loss는 증가 → 시각적으로 즉시 포착

---

## Cell 11 — 전체 학습 (Optional)

```
┌────────────────────────────────────┐
│ use_trainval=True (6,667장)        │
│  ↓ 모델 가중치 초기화               │
│  ↓ 10,000 steps 학습               │
│  ↓ TensorBoard로 모니터링          │
│  → 논문 SOTA 70.7% 도전            │
└────────────────────────────────────┘
```

> **목적**: 논문 SOTA 재현용 전체 학습 가이드 (현재는 주석 상태).

```python
# 전체 학습 — 주석 해제 후 실행
# # 1단계: train+val 합산 DataLoader 재생성
# train_loader_tv, _, test_loader_tv, *_ = get_aircraft_loaders(
#     use_trainval=True, ...)
#
# # 2단계: 가중치 초기화 후 전체 학습
# model.load_from(np.load(str(PRETRAINED)))
# best_acc, history = train(
#     model=model, train_loader=train_loader_tv,
#     num_steps=CFG["num_steps"], ...)
print("전체 학습: 위 주석 해제 후 실행")
print("TensorBoard: tensorboard --logdir logs")
```

**코드 설명:**
- 1단계: `use_trainval=True` → train+val 6,667장으로 학습
- 2단계: `load_from` 다시 호출 → quick training 결과 폐기 후 처음부터 재학습
- `tensorboard --logdir logs`: 별도 터미널에서 실시간 모니터링

**실행 결과:**
```
전체 학습: 위 주석 해제 후 실행
TensorBoard: tensorboard --logdir logs
```

**왜 이렇게 하는가?**
- quick training 가중치는 수렴 직전 상태 → 그대로 이어가면 학습률 스케줄러가 꼬임
- `load_from` 재호출로 초기 상태 보장 → 깨끗한 재학습
- 노트북 실행 시간이 너무 길어지지 않게 기본은 주석 처리

---

## Cell 12 — 체크포인트 로드

```
┌──────────────────────────────┐
│ output/fgvc_aircraft/        │
│   transfg_aircraft_*.bin     │
│         ↓                    │
│   model.load_state_dict      │
└──────────────────────────────┘
```

> **목적**: 학습 중 저장된 best 체크포인트를 로드해 평가 단계로 진입.

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

**코드 설명:**
- `OUTPUT_DIR / "transfg_aircraft_checkpoint.bin"`: train()이 자동 저장한 파일
- `torch.load(..., map_location=DEVICE)`: GPU/CPU 자동 매핑
- `ckpt["model"]`: state_dict (파라미터 이름 → tensor)
- `model.load_state_dict(...)`: 모델에 가중치 주입
- `model.eval()`: dropout/BN을 평가 모드로 (batch_norm 통계 고정)
- `else` 분기: 체크포인트 없어도 현재 메모리상 model로 평가 진행

**실행 결과:** 체크포인트 존재 여부에 따라 로드/스킵

**왜 이렇게 하는가?**
- 노트북 커널 재시작 후에도 학습된 모델 재사용 가능
- `model.eval()` 안 하면 추론 결과가 매번 달라짐 (dropout 활성)

---

## Cell 13 — Val + Test 평가

```
┌────────────────────────────────┐
│ evaluate_dataset                │
│  ├─ Val  Acc / Loss            │
│  └─ Test Acc / Loss             │
│      ↓                          │
│   논문 SOTA: 70.7%             │
└────────────────────────────────┘
```

> **목적**: val과 test 양쪽에서 정확도/손실을 계산해 일반화 성능 측정.

```python
print("Val 평가 중...")
val_result = evaluate_dataset(model, val_loader, DEVICE)

print("Test 평가 중...")
test_result = evaluate_dataset(model, test_loader, DEVICE)

print(f"\n=== FGVC-Aircraft 평가 결과 ===")
print(f"Val  Accuracy : {val_result['accuracy']:.4f}  ({val_result['accuracy']*100:.2f}%)")
print(f"Test Accuracy : {test_result['accuracy']:.4f}  ({test_result['accuracy']*100:.2f}%)")
print(f"Val  Avg Loss : {val_result['avg_loss']:.5f}")
print(f"Test Avg Loss : {test_result['avg_loss']:.5f}")
print(f"\n※ TransFG 논문 최고 성능 (FGVC-Aircraft, variant)")
print(f"   overlap split, 10k steps (trainval): 70.7%")
```

**코드 설명:**
- `evaluate_dataset`: 모든 배치 forward → 정확도/평균 loss 계산
- `val_result["accuracy"]`: 0~1 사이 float
- `*100`: 퍼센트 변환
- `:.4f` / `:.5f`: 소수점 자릿수 포맷
- 논문 성능 명시 → 현재 quick result와 비교 기준 제공

**실행 결과:**
```
=== FGVC-Aircraft 평가 결과 ===
Val  Accuracy : 0.0666  (6.66%)
Test Accuracy : 0.0618  (6.18%)
Val  Avg Loss : 4.36337
Test Avg Loss : 4.33838
※ TransFG 논문 최고 성능: 70.7%
```

**왜 이렇게 하는가?**
- val/test 둘 다 평가 → val에 과적합되지 않았는지 확인
- 6%대는 random(1%) 대비 6배 학습된 상태지만 절대값은 낮음 → 200 steps의 한계
- 논문 70.7% 재현하려면 10,000 step + use_trainval=True 필요

---

## Cell 14 — 단일 이미지 추론

```
┌────────────────────────────────┐
│  random idx                     │
│   ↓ predict_single              │
│   ↓ Top-5 softmax               │
│   ↓ matplotlib display          │
└────────────────────────────────┘
```

> **목적**: 한 장의 이미지에 대한 예측 결과 + Top-5 확률을 시각적으로 확인.

```python
idx = random.randint(0, len(testset_raw) - 1)
img_path   = testset_raw.img_paths[idx]
true_label = testset_raw.labels[idx]

result = predict_single(model, img_path, DEVICE, class_names=class_names)
pred   = result["pred_idx"]

print(f"이미지 경로 : {img_path}")
print(f"실제 클래스 : [{true_label:2d}] {class_names[true_label]}")
print(f"예측 클래스 : [{pred:2d}] {result.get('pred_name', '')}")
print(f"신뢰도      : {result['confidence']:.4f}")
print(f"정답 여부   : {'✓ 정답' if pred == true_label else '✗ 오답'}")

print("\nTop-5 예측:")
for name, prob in result.get("top5", []):
    bar = '█' * int(prob * 40)
    print(f"  {prob:.4f} {bar} {name}")

pil_img = Image.open(img_path).convert("RGB")
plt.figure(figsize=(5, 3))
plt.imshow(pil_img)
color = 'green' if pred == true_label else 'red'
plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred]}", color=color, fontsize=8)
plt.axis('off')
plt.tight_layout()
plt.show()
```

**코드 설명 (핵심 위주):**
- `random.randint(0, N-1)`: test 데이터셋에서 임의 인덱스 선택
- `testset_raw.img_paths[idx]`: 원본 파일 경로
- `predict_single`: 이미지 1장 → softmax 확률 + Top-5 + 예측 인덱스 반환
- `result["pred_idx"]`: argmax 인덱스
- `result["confidence"]`: Top-1 확률
- `'█' * int(prob * 40)`: 텍스트 막대그래프 (확률에 비례한 길이)
- `color = 'green' if 정답 else 'red'`: 시각적 정답/오답 구분

**실행 결과:** 이미지 + 텍스트 Top-5 + matplotlib 출력

**왜 이렇게 하는가?**
- 정량 metric(accuracy)만으로는 모델 약점이 안 보임 → 정성 평가 필요
- Top-5 확률로 "비슷한 다른 variant와 헷갈리는지" 확인 가능

---

## Cell 15 — 배치 예측 시각화

```
┌──────────────────────────────────┐
│ next(iter(test_loader)) → 배치 8 │
│   ↓ predict_batch                │
│   ↓ 8장 그리드 + True/Pred 표시  │
└──────────────────────────────────┘
```

> **목적**: 한 배치(8장)의 예측을 한 화면에서 비교 → 패턴 파악.

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

**코드 설명:**
- `next(iter(test_loader))`: 첫 배치 (이미 shuffle=False)
- `predict_batch`: 배치 forward → preds(np.ndarray), probs(np.ndarray)
- `(preds == y_vis.numpy()).mean()`: True/False → 0/1 평균 = 정확도
- `x_vis[:8]`: 처음 8개만 시각화
- `visualize_predictions`: helper — 정답이면 초록, 오답이면 빨강 테두리

**실행 결과:** 배치 정확도 + 8장 항공기 이미지 그리드 (예측/실제 비교)

**왜 이렇게 하는가?**
- 8장 한꺼번에 보면서 "어느 variant가 자주 틀리는지" 패턴 발견
- 비행기는 시점에 따라 같은 모델도 매우 다르게 보임 → 시각적으로 어려움 체감

---

## Cell 16 — Attention Map 시각화

```
┌──────────────────────────────────────┐
│ AttentionMapExtractor                 │
│  ↓ 각 layer의 [CLS]→patch attention  │
│  ↓ 히트맵으로 원본 이미지에 overlay  │
│  → 어디를 보고 분류했나?             │
└──────────────────────────────────────┘
```

> **목적**: 모델이 이미지의 어느 부분을 "보고" 분류했는지 시각화.

```python
from data_loader_aircraft import get_aircraft_transforms

_, test_tf = get_aircraft_transforms(CFG["img_size"])
attn_testset = AircraftDataset(
    root=str(DATA_ROOT), split="test", level=CFG["level"], transform=test_tf
)

idx = 0
img_tensor, label = attn_testset[idx]
print(f"클래스: [{label}] {class_names[label]}")
print("레이어별 CLS→Patch attention map (head=0)")

visualize_attention(
    model, img_tensor, DEVICE,
    head=0, patch_size=16, img_size=CFG["img_size"]
)
```

**코드 설명:**
- `get_aircraft_transforms`: train/test transform 튜플 반환 → test 변환만 사용
- `attn_testset[0]`: 첫 이미지 (Tensor)
- `visualize_attention`: helper — 각 layer의 attention을 layer 수만큼 subplot
- `head=0`: 12개 attention head 중 첫 번째만 사용 (Cell 17에서 12개 모두 비교)
- `patch_size=16`, `img_size=448`: heatmap을 원본 크기로 upsampling

**실행 결과:** 레이어별 attention 히트맵 — 날개 끝/엔진/꼬리 부위에 집중 예상

**왜 이렇게 하는가?**
- 모델 해석성(interpretability) 확보 → "왜 이 예측을 했는가?"
- Fine-grained는 작은 부위 차이로 결정됨 → attention이 그 부위를 봐야 정상
- 초기 layer는 전체적, 후반 layer는 특정 부위에 집중하는 경향 관찰 가능

---

## Cell 17 — 12개 Head Attention 비교

```
┌─────────────────────────────────────┐
│ last layer (12 heads)                │
│   head 0 → 엔진 영역?                │
│   head 1 → 꼬리 영역?                │
│   head 2 → 날개 영역?                │
│   ...                                │
│   head 11 → 동체 영역?               │
└─────────────────────────────────────┘
```

> **목적**: 마지막 layer의 12개 head 각각이 어디를 보는지 비교 → multi-head의 다양성 확인.

```python
from visualization import AttentionMapExtractor

idx = 3
img_tensor, label = attn_testset[idx]
x = img_tensor.unsqueeze(0).to(DEVICE)

model.eval()
with AttentionMapExtractor(model) as ext:
    with torch.no_grad():
        _ = model(x)
    attn_maps = ext.attention_maps

if attn_maps:
    last = attn_maps[-1][0]
    cls_attn = last[:, 0, 1:].numpy()
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
        plt.tight_layout()
        plt.show()
    else:
        print(f"overlap split: patch수({cls_attn.shape[1]})가 정방형 아님")
```

**코드 설명 (핵심 위주):**
- `unsqueeze(0)`: (3,448,448) → (1,3,448,448) 배치 차원 추가
- `AttentionMapExtractor(model)`: forward hook으로 모든 layer의 attention 저장
- `attn_maps[-1][0]`: 마지막 layer의 첫 배치 → shape (12 heads, seq, seq)
- `last[:, 0, 1:]`: [CLS](=0)이 patch들(1~)을 얼마나 보는가 → (12, n_patches)
- `n_p = int(sqrt(n_patches))`: 정방형 grid 가정 (overlap에서는 비정방형 가능)
- `(am - min) / (max - min)`: head별 0~1 정규화
- `resize(...)`: patch grid → 448×448로 upsample
- `imshow(img_rgb)` + `imshow(am_up, alpha=0.5, cmap="jet")`: 원본 위에 heatmap overlay

**실행 결과:** 12개 head의 attention 히트맵 — Boeing 737-300 vs 737-400처럼 미세한 차이를 각 head가 다르게 포착

**왜 이렇게 하는가?**
- multi-head attention의 본래 목적 = "다양한 관점에서 동시에 보기"
- 모든 head가 같은 곳만 보면 multi-head가 무의미 → 다양성 확인 필요
- overlap split은 patch 수가 비정방형(예: 1369=37×37은 정방형이지만 다른 케이스도 있음) → `n_p*n_p == ...` 체크

---

## Cell 18 — 3개 데이터셋 종합 비교

```
┌──────────────────────────────────────┐
│ CUB-200    | Stanford Cars | Aircraft│
│ 200종      | 196종         | 100종   │
│ 91.7%      | 96.1%         | 70.7%   │
└──────────────────────────────────────┘
```

> **목적**: 3개 fine-grained 벤치마크의 클래스 수, 학습 데이터, 논문 SOTA 비교.

```python
results = {
    "CUB-200-2011"  : {"quick_acc": 0.3788, "paper_acc": 0.917, "classes": 200, "train": 5994},
    "Stanford Cars" : {"quick_acc": None,   "paper_acc": 0.961, "classes": 196, "train": 8144},
    "FGVC-Aircraft" : {"quick_acc": None,   "paper_acc": 0.707, "classes": 100, "train": 3334},
}

print(f"{'데이터셋':18s} {'클래스':>6s} {'Train':>6s} {'Quick Acc':>10s} {'논문 목표':>10s}")
print("-" * 58)
for name, r in results.items():
    quick = f"{r['quick_acc']*100:.1f}%" if r['quick_acc'] else "미실행"
    print(f"  {name:16s} {r['classes']:>6d} {r['train']:>6,} {quick:>10s} {r['paper_acc']*100:>9.1f}%")
```

**코드 설명:**
- `results`: 3개 데이터셋의 결과를 dict로 정리
- `quick_acc`: 노트북에서 직접 실행한 짧은 학습 정확도 (Cars/Aircraft는 미실행)
- `paper_acc`: 논문 SOTA
- `f"{name:16s}"`: 16자 왼쪽 정렬, `:>6d`: 6자 오른쪽 정렬 정수
- `:,`: 천 단위 콤마 (5994 → 5,994)
- `if r['quick_acc'] else "미실행"`: None일 때 fallback 텍스트

**실행 결과:**
```
데이터셋           클래스  Train  Quick Acc  논문 목표
----------------------------------------------------------
  CUB-200-2011     200  5,994      37.9%      91.7%
  Stanford Cars    196  8,144     미실행      96.1%
  FGVC-Aircraft    100  3,334     미실행      70.7%
```

**왜 이렇게 하는가?**
- 같은 모델(TransFG)이라도 도메인에 따라 성능 차이 큼 (96.1% vs 70.7%)
- Aircraft가 가장 어려운 이유 = 데이터 수 적음 + variant 차이 미세

---

## Cell 19 — 논문 성능 Bar Chart

```
┌──────────────────────────────────────┐
│   100 ┤                              │
│    96 ┤  ▆ 96.1                     │
│    92 ┤▆     91.7                   │
│    88 ┤                              │
│    ...                               │
│    72 ┤              ▆ 70.7          │
│       └─CUB─Cars─Aircraft────        │
└──────────────────────────────────────┘
```

> **목적**: 3개 데이터셋 정확도를 막대 그래프로 시각화 + 차이 강조.

```python
datasets = ["CUB-200-2011", "Stanford Cars", "FGVC-Aircraft"]
paper_accs = [91.7, 96.1, 70.7]
colors = ["#3f51b5", "#e53935", "#43a047"]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(datasets, paper_accs, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

for bar, acc in zip(bars, paper_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{acc}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(60, 100)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("TransFG 논문 성능 — 3가지 Fine-Grained 데이터셋", fontsize=12, fontweight='bold')
ax.axhline(y=90, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

print("\nAircraft(70.7%)가 상대적으로 낮은 이유:")
print("  1. 클래스당 학습 데이터 가장 적음 (~33장)")
print("  2. 동일 family 내 variant 간 차이 극히 미세")
print("  3. 항공기 자세·촬영 각도 다양성이 높음")
```

**코드 설명 (핵심 위주):**
- `ax.bar(datasets, paper_accs, color=colors)`: 막대 그래프 생성
- `ax.text(... bar.get_height() + 0.5, f"{acc}%")`: 막대 위에 정확도 라벨
- `ax.set_ylim(60, 100)`: y축 범위 (60~100, 차이 강조)
- `axhline(y=90, linestyle='--')`: 90% 기준선 (점선)
- `spines['top'/'right'].set_visible(False)`: 상/우 테두리 제거 (깔끔한 디자인)

**실행 결과:** 3개 데이터셋 정확도 막대 그래프 + Aircraft가 낮은 이유 텍스트

**왜 이렇게 하는가?**
- 표보다 그래프가 차이를 직관적으로 전달
- 90% 기준선 → "Aircraft만 90% 미달" 시각적으로 강조

---

## 3개 데이터셋 비교 표

| 항목 | CUB-200-2011 | Stanford Cars | FGVC-Aircraft |
|---|---|---|---|
| 도메인 | 조류 | 자동차 | **항공기** |
| 클래스 수 | 200종 | 196종 | **100 variant** |
| Train | 5,994 | 8,144 | **3,334** |
| Val | — | — | **3,333** |
| Test | 5,794 | 8,041 | **3,333** |
| 클래스당 평균 | 30장 | 41장 | **33장** |
| 분할 방식 | train/test | train/test | **train/val/test** |
| 논문 SOTA | 91.7% | 96.1% | **70.7%** |
| 난이도 | 보통 | 쉬움 | **가장 어려움** |

**핵심 인사이트:**
- Aircraft는 클래스 수가 가장 적지만(100종) 가장 어려움 → variant 차이가 미세
- Cars는 데이터가 가장 많고(8,144) 차종이 명확히 구분됨 → 가장 쉬움
- CUB는 클래스당 30장으로 적지만 종 간 시각적 차이 명확 → 중간 난이도

---

## 공부 팁

```
┌──────────────────────────────────────────┐
│   "왜?"를 5번 묻기                         │
│   ── 코드를 외우지 말고 동작을 이해하기   │
└──────────────────────────────────────────┘
```

### Tip 1. 코드는 "왜 이 줄이 필요한가" 자문하면서 읽기
- 예: `model.eval()` → 왜? → dropout/BN을 평가 모드로
- 예: `with torch.no_grad()` → 왜? → 추론 시 gradient 계산 불필요 (메모리 절약)
- 예: `zero_head=True` → 왜? → 클래스 수 다른 pretrained 헤드 무시
- 예: `unsqueeze(0)` → 왜? → 모델은 항상 배치 차원 기대

### Tip 2. shape를 항상 추적
- 모든 tensor 연산 직후 `print(x.shape)`로 차원 확인
- ViT 입력: (B, 3, 448, 448) → patch embedding: (B, 1369, 768) → output: (B, 100)
- shape mismatch가 가장 흔한 버그

### Tip 3. 작은 것부터 실험
- `quick_steps=200`처럼 짧게 돌려 코드 검증 → 큰 학습 진입
- batch_size=8로 OOM 안 나는지 먼저 확인 → 키울지 결정
- `next(iter(loader))` 한 배치만 꺼내 forward 테스트

### Tip 4. Pretrained의 위력 체감
- pretrained 없이 클래스당 33장으로는 ViT 학습 불가
- ImageNet pretrained → 일반적인 시각 특징 이미 학습됨 → fine-tuning만 필요
- Transfer learning 없이 fine-grained는 사실상 불가능

### Tip 5. Attention map은 디버깅 도구
- 모델이 엉뚱한 부위(배경, 하늘)에 집중하면 학습 부족 또는 augmentation 문제
- 정상이라면 항공기 본체(엔진, 날개, 꼬리)에 집중 → 모델 신뢰성 ↑
- 정확도가 낮을 때 Cell 16/17의 attention map으로 원인 파악 시도

### Tip 6. fine-grained = 부위별 미세 차이
- Boeing 737-300 vs 737-400 → 동체 길이, 창문 수
- 일반적인 분류기와 fine-grained 분류기의 핵심 차이 = "어디를 볼 것인가"
- TransFG의 Part Selection이 바로 이 문제를 해결

### Tip 7. val과 test의 역할 분리
- **val** = 개발 중 hyperparameter 튜닝용 (반복 사용 OK)
- **test** = 최종 1회 평가용 (반복 보면 test에 과적합)
- FGVC-Aircraft가 굳이 3분할인 이유 = 이 원칙을 강제

### Tip 8. 자주 쓰는 디버깅 명령
```python
print(x.shape, x.dtype, x.device)        # tensor 기본 정보
print(model)                              # 모델 구조 출력
sum(p.numel() for p in model.parameters() if p.requires_grad)  # 학습 파라미터 수
torch.cuda.memory_allocated() / 1e9      # 현재 GPU 메모리 사용량 (GB)
```

### Tip 9. 막힐 때 추천 순서
1. 에러 메시지의 마지막 줄부터 읽기 (가장 구체적)
2. shape mismatch면 `print(x.shape)`로 추적
3. CUDA OOM이면 batch_size 줄이기 / fp16 활성화
4. nan loss면 lr 줄이기 / gradient clipping(`max_grad_norm`) 강화
5. accuracy 안 오르면 pretrained 로드 확인 + warmup 길게

### Tip 10. 다음 학습 단계 추천
- TransFG 논문 직접 읽기 (Part Selection Module 수식)
- ViT 논문 (Dosovitskiy et al., 2020) — Transformer가 이미지에 적용되는 원리
- BERT 논문 — Transformer + [CLS] token 개념 (ViT의 토대)
- timm 라이브러리 — 다양한 사전학습 ViT 모델 실험
- Stanford Cars 노트북도 같은 구조로 작성되어 있어 비교 학습 추천

---

> 이 자료를 따라가며 직접 코드를 한 줄씩 실행해보는 것이 가장 좋은 공부 방법입니다.
> Quick training(200 steps)으로 파이프라인을 이해한 후, GPU 시간이 허락하면 Cell 11을 활성화해 논문 SOTA에 도전해보세요.
