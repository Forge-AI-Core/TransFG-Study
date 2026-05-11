# TransFG 완전 학습 가이드
## 파이썬 기초 수준에서 Fine-Grained Recognition 논문까지

> **대상**: 파이썬 기초 문법을 아는 대학 1학년  
> **목표**: TransFG 논문을 이해하고, CUB-200-2011 데이터셋으로 새 종류를 분류하는 AI를 직접 학습시킨다

---

## 📚 목차

1. [들어가기 전에 — 큰 그림 이해하기](#1-들어가기-전에--큰-그림-이해하기)
2. [배경 지식 — Transformer란 무엇인가](#2-배경-지식--transformer란-무엇인가)
3. [CUB-200-2011 데이터셋](#3-cub-200-2011-데이터셋)
4. [프로젝트 파일 구조](#4-프로젝트-파일-구조)
5. [TransFG 전체 아키텍처 흐름](#5-transfg-전체-아키텍처-흐름)
6. [코드 파일별 상세 설명](#6-코드-파일별-상세-설명)
   - [dataset\_cub\_fixed.py](#61-dataset_cub_fixedpy--데이터셋-클래스)
   - [data\_loader.py](#62-data_loaderpy--데이터-전처리--로더)
   - [TransFG/models/configs.py](#63-transfgmodelsconfigspy--모델-설정)
   - [TransFG/models/modeling.py](#64-transfgmodelsmodelingpy--모델-핵심-코드)
   - [trainer.py](#65-trainerpy--학습-엔진)
   - [inference\_utils.py](#66-inference_utilspy--추론)
   - [visualization.py](#67-visualizationpy--시각화)
7. [모델 아키텍처 깊이 파기](#7-모델-아키텍처-깊이-파기)
   - [Patch Embedding](#71-patch-embedding--이미지를-단어처럼)
   - [Position Embedding](#72-position-embedding--순서-정보-추가)
   - [Multi-Head Self-Attention](#73-multi-head-self-attention--가장-중요한-부분)
   - [MLP Block](#74-mlp-block--정보-처리)
   - [Part Selection Module (PSM)](#75-part-selection-module-psm--transfg의-핵심)
   - [Contrastive Loss](#76-contrastive-loss--같은-종끼리-모아라)
8. [학습 전략](#8-학습-전략)
9. [노트북 셀별 가이드](#9-노트북-셀별-가이드)
10. [자주 묻는 질문](#10-자주-묻는-질문-faq)
11. [핵심 용어 사전](#11-핵심-용어-사전)

---

## 1. 들어가기 전에 — 큰 그림 이해하기

### 1.1 이 프로젝트가 하려는 것

```
[새 사진 한 장 입력]
        ↓
[TransFG 모델]
        ↓
"이것은 'American Goldfinch' 입니다 (신뢰도: 94.3%)"
```

**200종류의 새**를 사진만 보고 구분하는 AI를 만드는 것입니다.

### 1.2 왜 어려운가? — Fine-Grained Recognition

일반적인 이미지 분류는 "개 vs 고양이 vs 자동차"처럼 **외형이 크게 다른** 것들을 구분합니다.  
하지만 Fine-Grained (세밀 분류)는 **같은 종류 안에서 세부 구분**을 합니다.

```
일반 분류:                    Fine-Grained 분류:
┌────────┐                   ┌─────────────────┐
│  🐦   │ → "새"            │  🐦 (흰 가슴)   │ → "White Pelican"
│       │                   │  🐦 (검은 날개) │ → "Black Skimmer"  
│  🐕   │ → "개"            │  🐦 (붉은 머리) │ → "Red Kite"
└────────┘                   └─────────────────┘
차이가 크다 → 쉬움            차이가 미묘하다 → 어렵다!
```

AI가 새의 **부리 모양**, **날개 무늬**, **발 색깔** 같은 **미세한 차이**를 찾아야 합니다.

### 1.3 TransFG의 아이디어

**TransFG = Transformer + Fine-Grained**

> "사람도 새를 구분할 때 전체를 보지 않고, 특징적인 부위(부리, 머리 색 등)에 집중한다.  
> AI도 마찬가지로 중요한 부위에 집중하게 만들자!"

---

## 2. 배경 지식 — Transformer란 무엇인가

### 2.1 CNN (Convolutional Neural Network) — 기존 방법

CNN은 이미지를 **작은 필터로 훑어가며** 특징을 추출합니다.

```
[이미지]  →  [필터1] →  [필터2] →  ... →  [분류]
           특선선 감지   윤곽 감지      고수준 특징
```

**한계**: CNN은 **주변 픽셀만** 봅니다. 이미지의 왼쪽 위와 오른쪽 아래를 동시에 연결하기 어렵습니다.

### 2.2 Transformer — 새로운 방법 (원래 번역을 위해 만들어짐)

원래 자연어처리(NLP, 텍스트 처리)를 위해 만들어졌습니다.

```
문장: "나는 사과를 먹었다"
       ↓
각 단어를 숫자 벡터로 변환
"나는"→[0.2, 0.8, ...], "사과를"→[0.5, 0.1, ...], ...
       ↓
모든 단어가 서로를 봄 (Attention!)
"먹었다"는 "사과를"을 주목 → 관계 파악
```

### 2.3 ViT (Vision Transformer) — 이미지에 Transformer 적용

**아이디어**: 이미지를 **패치(조각)**로 잘라서 단어처럼 취급하자!

```
원본 이미지 (448×448 픽셀)
┌─────────────────────────┐
│  🐦  이미지             │
│                         │
│    16×16 패치로 분할    │
│                         │
└─────────────────────────┘
           ↓
패치들: [P1][P2][P3]...[P784]
(28×28 = 784개 패치)
           ↓
각 패치 = 하나의 "단어" 처럼 처리
```

### 2.4 Attention (어텐션) — 핵심 개념

> **"어디에 집중할 것인가?"** 를 배우는 메커니즘

예시: "새의 부리를 봐라"라는 힌트 없이도, 모델이 스스로 부리가 중요하다는 것을 학습합니다.

```
모든 패치 쌍에 대해:
P1 ↔ P2: 서로 얼마나 관련 있나? → 0.2 (별로)
P1 ↔ P45: 서로 얼마나 관련 있나? → 0.9 (매우 관련!)
P1 ↔ P234: 서로 얼마나 관련 있나? → 0.05 (무관)

→ 높은 점수 = 더 주목! (Attention)
```

---

## 3. CUB-200-2011 데이터셋

### 3.1 데이터셋 소개

- **CUB** = Caltech-UCSD Birds
- **200** = 200종의 새
- **2011** = 2011년 공개
- **11,788장** 이미지 (Train: 5,994장, Test: 5,794장)

### 3.2 디렉토리 구조

```
data/CUB_200_2011/
│
├── images/                          ← 실제 이미지 폴더
│   ├── 001.Black_footed_Albatross/
│   │   ├── Black_Footed_Albatross_0001_796111.jpg
│   │   ├── Black_Footed_Albatross_0002_55.jpg
│   │   └── ... (약 60장)
│   ├── 002.Laysan_Albatross/
│   │   └── ...
│   └── ... (200개 폴더)
│
├── images.txt                       ← 이미지 번호 ↔ 파일명 매핑
│   # 형식: "이미지번호 파일경로"
│   # 예: "1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
│
├── image_class_labels.txt           ← 이미지 번호 ↔ 클래스 번호
│   # 예: "1 1" (이미지1은 클래스1)
│
├── train_test_split.txt             ← 훈련/테스트 구분
│   # 예: "1 1" (이미지1은 훈련용)
│   # 예: "2 0" (이미지2는 테스트용)
│
├── classes.txt                      ← 클래스 번호 ↔ 종 이름
│   # 예: "1 001.Black_footed_Albatross"
│
└── bounding_boxes.txt               ← 새가 있는 위치 (좌표)
```

### 3.3 클래스 예시 (200종 중 일부)

```
클래스  1: Black_footed_Albatross    (검은발 알바트로스)
클래스 10: Red_winged_Blackbird      (붉은 날개 찌르레기)
클래스 50: Acadian_Flycatcher        (아카디아 딱새)
클래스100: Brown_Pelican              (갈색 사다새)
클래스150: Yellow_headed_Blackbird   (노랑머리 찌르레기)
클래스200: Common_Yellowthroat        (황금 목 Common)
```

---

## 4. 프로젝트 파일 구조

```
AiffelThon01/
│
├── 📔 TransFG_study.ipynb         ← 메인 노트북 (여기서 실험!)
├── 📖 STUDY_GUIDE.md              ← 지금 읽고 있는 이 파일
│
├── 🐍 dataset_cub_fixed.py        ← CUB 데이터셋 읽기 (수정된 버전)
├── 🐍 data_loader.py              ← 이미지 전처리 + DataLoader
├── 🐍 trainer.py                  ← 학습 루프 (training loop)
├── 🐍 inference_utils.py          ← 추론 (예측하기)
├── 🐍 visualization.py            ← 결과 시각화
├── 📄 requirements_cu130.txt      ← 필요 패키지 목록
│
├── TransFG/                       ← 원본 논문 코드
│   ├── 🐍 train.py                ← 원본 학습 스크립트 (멀티GPU용)
│   ├── models/
│   │   ├── 🐍 modeling.py         ← 모델 핵심 코드 ⭐
│   │   └── 🐍 configs.py          ← 모델 설정값
│   └── utils/
│       ├── 🐍 scheduler.py        ← 학습률 스케줄러
│       ├── 🐍 data_utils.py       ← 데이터 로더 (원본)
│       ├── 🐍 dataset.py          ← 데이터셋 (원본, 버그 있음)
│       └── 🐍 autoaugment.py      ← 데이터 증강
│
├── data/
│   └── CUB_200_2011/              ← 데이터셋
├── pretrained/
│   └── ViT-B_16.npz               ← ImageNet-21k 사전학습 가중치
├── output/
│   └── transfg_cub_checkpoint.bin ← 학습된 모델 (학습 후 생성)
├── logs/                          ← TensorBoard 로그
└── TransFG_venv/                  ← 가상환경
```

---

## 5. TransFG 전체 아키텍처 흐름

### 5.1 큰 그림 — 입력에서 출력까지

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

새 이미지 1장 (448 × 448 × 3)
   [R,G,B 채널이 3개인 448픽셀짜리 정사각형 이미지]

   ↓ ① PATCH EMBEDDING
   
784개 패치 + 1개 CLS 토큰 = 785개 벡터
   각 벡터의 크기: 768차원
   [CLS][P1][P2][P3]...[P784]  →  shape: (785, 768)

   ↓ ② POSITION EMBEDDING 추가
   
   위치 정보 추가 (어떤 패치가 어디 있었는지)

   ↓ ③ TRANSFORMER ENCODER × 11 (Block 1~11)
   
   각 블록에서:
   - Self-Attention: 모든 패치가 서로를 본다
   - MLP: 정보를 변환한다
   - Attention Weights(가중치) 수집 → 나중에 사용!

   ↓ ④ PART SELECTION (PSM) — TransFG의 핵심!
   
   11개 블록의 Attention 가중치를 곱해서
   "가장 중요한 패치"를 선택
   예: P23, P156, P445, P567, P678 (새의 부리, 눈 위치)

   ↓ ⑤ FINAL TRANSFORMER BLOCK (Block 12)
   
   CLS + 선택된 중요 패치들 → 최종 표현

   ↓ ⑥ CLASSIFICATION HEAD
   
   768차원 → 200차원 (200종 점수)
   [0.01, 0.03, ..., 0.94, ..., 0.02]
   
   ↓ ⑦ SOFTMAX → 확률
   
   "American Goldfinch: 94%" ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.2 학습과 추론의 차이

| 구분 | 학습 (Training) | 추론 (Inference) |
|------|----------------|-----------------|
| 입력 | 이미지 + 정답 레이블 | 이미지만 |
| 출력 | 손실값(Loss) + 예측 | 예측 (클래스) |
| 목적 | 가중치를 업데이트 | 예측만 수행 |
| `model(x, y)` 호출 | `model(x, labels=y)` | `model(x)` |

---

## 6. 코드 파일별 상세 설명

---

### 6.1 `dataset_cub_fixed.py` — 데이터셋 클래스

**역할**: CUB-200-2011 폴더를 읽어서 "이미지 경로 + 클래스 번호" 쌍을 만든다.

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일: dataset_cub_fixed.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import numpy as np
from PIL import Image               # 이미지 읽기 라이브러리
from torch.utils.data import Dataset # PyTorch 데이터셋 기반 클래스

class CUBDataset(Dataset):
    """
    Dataset = PyTorch가 이해하는 데이터 형태
    
    필수로 구현해야 하는 메서드:
    - __len__  : 전체 데이터 수를 반환
    - __getitem__: 인덱스로 데이터 하나를 반환
    """

    def __init__(self, root: str, is_train: bool = True, transform=None):
        """
        root     : 데이터 폴더 경로 (예: "data/CUB_200_2011")
        is_train : True면 훈련용, False면 테스트용
        transform: 이미지에 적용할 변환(전처리) 함수
        """
        self.root = root
        self.is_train = is_train
        self.transform = transform

        # ── 텍스트 파일 읽기 ──────────────────────────────────
        # images.txt 읽기: 각 줄의 마지막 단어(파일명)만 추출
        img_names = self._read_lines(
            os.path.join(root, "images.txt"), col=-1
        )
        # 예: ['001.Black_footed.../Black_Footed_0001.jpg', ...]
        
        # image_class_labels.txt 읽기: 클래스 번호 (1~200) → 0~199로 변환
        labels = [
            int(v) - 1  # "-1" 이유: 파일은 1부터 시작, Python은 0부터
            for v in self._read_lines(
                os.path.join(root, "image_class_labels.txt"), col=-1
            )
        ]
        
        # train_test_split.txt 읽기: 1=훈련, 0=테스트
        splits = [
            int(v) for v in self._read_lines(
                os.path.join(root, "train_test_split.txt"), col=-1
            )
        ]

        # ── 훈련/테스트 분리 ──────────────────────────────────
        pairs = list(zip(splits, img_names, labels))
        # pairs = [(1, '001.../0001.jpg', 0), (0, '001.../0002.jpg', 0), ...]
        
        if is_train:
            pairs = [(n, l) for s, n, l in pairs if s == 1]  # split==1인것만
        else:
            pairs = [(n, l) for s, n, l in pairs if s == 0]  # split==0인것만

        # 경로 조합: root/images/001.Black.../0001.jpg
        self.img_paths = [os.path.join(root, "images", n) for n, _ in pairs]
        self.labels = [l for _, l in pairs]

    @staticmethod
    def _read_lines(path: str, col: int) -> list:
        """텍스트 파일 한 줄씩 읽어서 col번째 단어만 추출"""
        with open(path) as f:
            return [line.strip().split()[col] for line in f]

    def __len__(self):
        """전체 샘플 수 반환 — PyTorch가 자동으로 호출"""
        return len(self.labels)

    def __getitem__(self, index):
        """
        index번째 샘플 반환 — 핵심 메서드!
        dataset[0], dataset[100] 처럼 인덱싱할 때 호출됨
        """
        # 이미지 읽기 (PIL = Python Imaging Library)
        img = Image.open(self.img_paths[index]).convert("RGB")
        # .convert("RGB") → 그레이스케일이나 RGBA 이미지를 RGB로 통일
        
        # 변환 적용 (리사이즈, 정규화 등)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.labels[index]  # (이미지 텐서, 정수 레이블)

    def get_class_names(self) -> list:
        """classes.txt에서 클래스 이름 목록 반환"""
        path = os.path.join(self.root, "classes.txt")
        names = {}
        with open(path) as f:
            for line in f:
                idx, name = line.strip().split(maxsplit=1)
                names[int(idx) - 1] = name  # 0-indexed
        return [names[i] for i in range(len(names))]
```

**원본과의 차이점:**
```python
# ❌ 원본 (utils/dataset.py) — scipy.misc.imread는 제거된 함수!
self.train_img = [scipy.misc.imread(os.path.join(...)) for ...]
# 모든 이미지를 RAM에 한 번에 로드 → 메모리 부족 가능

# ✅ 수정 버전 (dataset_cub_fixed.py) — 경로만 저장, 실제 로딩은 나중에!
self.img_paths = [os.path.join(...) for ...]  # 경로만 저장
# → __getitem__ 호출 시마다 필요한 이미지만 읽음 (Lazy Loading)
```

---

### 6.2 `data_loader.py` — 데이터 전처리 + 로더

**역할**: 이미지를 모델이 받을 수 있는 형태로 변환하고, 배치(묶음)로 만든다.

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일: data_loader.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from PIL import Image
from torchvision import transforms    # 이미지 변환 도구 모음
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# ── Pillow 버전 호환성 처리 ──────────────────────
try:
    _BILINEAR = Image.Resampling.BILINEAR  # Pillow 10+ (최신)
except AttributeError:
    _BILINEAR = Image.BILINEAR             # Pillow 9 이하 (구버전)
# 이유: Pillow 10.0에서 Image.BILINEAR가 제거됨
#       AttributeError = "그 속성이 없다"는 에러
```

**변환 파이프라인 설명:**

```python
def get_cub_transforms(img_size: int = 448):
    """
    이미지 전처리 변환 설정
    
    왜 이렇게 복잡하게 하나?
    → 학습 시: 다양한 변형으로 "데이터 증강" → 과적합 방지
    → 테스트 시: 일관된 변환으로 공정한 평가
    """
    
    train = transforms.Compose([
        # ① 600×600으로 확대 (원본 크기 다양: 100~1500픽셀)
        transforms.Resize((600, 600), _BILINEAR),
        
        # ② 448×448 랜덤 크롭 (매번 다른 위치에서 자름)
        #    → 같은 이미지도 매 에포크마다 다르게 보임 (증강!)
        transforms.RandomCrop((448, 448)),
        
        # ③ 50% 확률로 좌우 반전 (또 다른 증강)
        transforms.RandomHorizontalFlip(),
        
        # ④ PIL 이미지 → PyTorch 텐서 변환
        #    (H, W, C) → (C, H, W), 값 범위: 0~255 → 0.0~1.0
        transforms.ToTensor(),
        
        # ⑤ ImageNet 통계로 정규화
        #    mean=[0.485, 0.456, 0.406] = R,G,B 평균
        #    std =[0.229, 0.224, 0.225] = R,G,B 표준편차
        #    공식: (픽셀값 - mean) / std
        #    → 모델 학습이 안정적으로 됨
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test = transforms.Compose([
        transforms.Resize((600, 600), _BILINEAR),
        # 테스트는 랜덤 크롭 대신 중앙 크롭 (항상 동일한 영역)
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    return train, test
```

**왜 448×448인가?** (크기 선택 이유)
```
ViT-B/16: 패치 크기 = 16×16
448 / 16 = 28  → 28×28 = 784개 패치 (매우 상세)

만약 224×224라면: 224/16 = 14 → 14×14 = 196개 패치 (덜 상세)
→ TransFG는 세밀한 특징을 보려고 높은 해상도 사용
```

**DataLoader란?**
```python
train_loader = DataLoader(
    trainset,                    # 우리가 만든 CUBDataset
    sampler=RandomSampler(trainset),  # 훈련: 무작위 순서로 뽑기
    batch_size=8,                # 한 번에 8장씩 묶어서
    num_workers=4,               # 4개 CPU 코어가 미리 준비
    drop_last=True,              # 마지막 불완전한 배치 버리기
    pin_memory=True,             # GPU 전송 속도 향상
)

# DataLoader 사용:
for x, y in train_loader:
    # x: shape (8, 3, 448, 448) — 8장의 이미지 텐서
    # y: shape (8,)             — 8개의 정수 레이블
    ...
```

---

### 6.3 `TransFG/models/configs.py` — 모델 설정

**역할**: ViT 모델의 크기와 구조를 결정하는 설정값 모음

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일: TransFG/models/configs.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import ml_collections  # 딕셔너리처럼 쓰는 설정 클래스

def get_b16_config():
    """ViT-B/16 설정 — B=Base, 16=패치크기"""
    config = ml_collections.ConfigDict()
    
    config.patches = ml_collections.ConfigDict({
        'size': (16, 16)    # 패치 크기: 16×16 픽셀
    })
    config.split = 'non-overlap'  # 패치 방식 (나중에 'overlap'으로 바꿈)
    config.slide_step = 12        # overlap일 때 이동 간격
    config.hidden_size = 768      # 각 벡터(토큰)의 크기
    
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072         # MLP 내부 크기 (hidden_size × 4)
    config.transformer.num_heads = 12         # Attention Head 수
    config.transformer.num_layers = 12        # Transformer Block 수
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    
    config.classifier = 'token'  # CLS 토큰으로 분류
    config.representation_size = None
    
    return config
```

**크기 비교:**

| 모델 | hidden_size | num_layers | num_heads | 파라미터 |
|------|------------|------------|-----------|---------|
| ViT-B/16 | 768 | 12 | 12 | ~86M |
| ViT-L/16 | 1024 | 24 | 16 | ~307M |
| ViT-H/14 | 1280 | 32 | 16 | ~632M |

```
우리가 사용하는 ViT-B/16:
- hidden_size=768: 각 패치 = 768개의 숫자로 표현
- num_layers=12: Transformer 블록이 12개 쌓임
- num_heads=12: Attention을 12개의 시각으로 동시에 봄
- 파라미터 ~86M: 86백만 개의 학습 가능한 숫자
```

---

### 6.4 `TransFG/models/modeling.py` — 모델 핵심 코드

**이 파일이 TransFG의 핵심입니다.** 여러 클래스로 구성됩니다.

#### 전체 클래스 구조

```
VisionTransformer (최상위 모델)
    └── Transformer
         ├── Embeddings (패치 임베딩 + 위치 임베딩)
         │    └── Conv2d (패치를 벡터로 변환)
         └── Encoder
              ├── Block × 11 (Transformer Block × 11)
              │    ├── Attention (Multi-Head Self-Attention)
              │    │    ├── Linear (query, key, value, out)
              │    │    └── Softmax, Dropout
              │    ├── Mlp (Feed-Forward Network)
              │    │    └── Linear × 2
              │    └── LayerNorm × 2
              ├── Part_Attention (PSM — TransFG 핵심!)
              ├── Block (마지막 Block 12)
              └── LayerNorm
```

#### ① `np2th()` 함수 — 가중치 형식 변환

```python
def np2th(weights, conv=False):
    """
    NumPy 배열 → PyTorch 텐서 변환
    
    왜 필요한가?
    Google의 ViT는 JAX/TensorFlow로 학습됨
    JAX 가중치는 HWIO 형식 (Height, Width, In, Out)
    PyTorch는 OIHW 형식 (Out, In, Height, Width)
    → conv=True면 형식 변환 필요
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])  # HWIO → OIHW
    return torch.from_numpy(weights)
```

#### ② `Attention` 클래스 — 가장 중요한 부분!

```python
class Attention(nn.Module):
    """
    Multi-Head Self-Attention
    
    "Self" = 입력 자기 자신과 비교
    "Multi-Head" = 여러 시각(Head)으로 동시에 봄
    """
    
    def __init__(self, config):
        super().__init__()
        # config.hidden_size = 768
        # config.transformer["num_heads"] = 12
        
        self.num_attention_heads = 12
        self.attention_head_size = 768 // 12  # = 64 (각 헤드의 크기)
        self.all_head_size = 12 * 64          # = 768
        
        # Q, K, V 선형 변환 (각각 768→768)
        # Query: "무엇을 찾고 있나?"
        # Key:   "나는 무엇인가?"
        # Value: "내가 전달할 정보는?"
        self.query = Linear(768, 768)
        self.key   = Linear(768, 768)
        self.value = Linear(768, 768)
        
        self.out = Linear(768, 768)  # 출력 변환
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)
        self.softmax = Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        """
        (배치, 시퀀스, 768) → (배치, 12헤드, 시퀀스, 64)
        12개의 헤드로 분리
        """
        new_x_shape = x.size()[:-1] + (12, 64)
        # (B, seq_len, 768) → (B, seq_len, 12, 64)
        x = x.view(*new_x_shape)
        # (B, seq_len, 12, 64) → (B, 12, seq_len, 64)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        """
        hidden_states: (배치, 785, 768)
        - 785 = CLS(1) + 패치(784)
        - 768 = 각 토큰의 벡터 크기
        """
        # ① Q, K, V 계산
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))
        # 각각 shape: (B, 12, 785, 64)
        
        # ② Attention Score 계산
        # Q × K^T: 각 토큰 쌍의 유사도 계산
        # shape: (B, 12, 785, 785)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        
        # ③ 스케일링 (√64 = 8로 나눔)
        # 왜? 차원이 크면 내적값이 너무 커져 gradient가 사라지는 문제 방지
        attention_scores = attention_scores / math.sqrt(64)
        
        # ④ Softmax → 확률(합=1)로 변환 = Attention Weight
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs  # ← 나중에 PSM에서 사용!
        
        # ⑤ V에 가중합 적용
        # shape: (B, 12, 785, 64)
        context_layer = torch.matmul(attention_probs, V)
        
        # ⑥ 헤드 합치기: (B, 12, 785, 64) → (B, 785, 768)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-2] + (768,)
        )
        
        # ⑦ 출력 선형 변환
        attention_output = self.out(context_layer)
        
        return attention_output, weights  # weights가 중요!
```

**Attention 메커니즘 시각화:**

```
입력: [CLS][P1][P2][P3]...[P784]  (785개 토큰)

            CLS  P1   P2  ...  P784
           ┌────────────────────────┐
       CLS │0.01 0.02 0.15 ... 0.03│  ← CLS가 각 패치를 얼마나 주목?
        P1 │0.08 0.00 0.02 ... 0.01│
        P2 │0.12 0.03 0.00 ... 0.02│
       ... │  ...                  │
       P784│0.04 0.01 0.01 ... 0.00│
           └────────────────────────┘
               Attention Matrix (785×785)

CLS→P2 = 0.15 (높음) → P2 패치가 중요하다!
```

#### ③ `Mlp` 클래스 — 정보 변환

```python
class Mlp(nn.Module):
    """
    Feed-Forward Network (FFN) / MLP
    
    Attention으로 수집한 정보를 비선형 변환으로 처리
    """
    def __init__(self, config):
        super().__init__()
        # 768 → 3072 → 768 (확장 후 축소)
        self.fc1 = Linear(768, 3072)   # 4배 확장
        self.fc2 = Linear(3072, 768)   # 다시 원래 크기로
        self.act_fn = torch.nn.functional.gelu  # 활성화 함수
        self.dropout = Dropout(0.1)
    
    def forward(self, x):
        x = self.fc1(x)      # (B, 785, 768) → (B, 785, 3072)
        x = self.act_fn(x)   # 비선형성 추가 (GELU)
        x = self.dropout(x)
        x = self.fc2(x)      # (B, 785, 3072) → (B, 785, 768)
        x = self.dropout(x)
        return x
```

**왜 확장했다가 다시 줄이나?**
```
768 → 3072 → 768

고차원으로 확장: 더 복잡한 표현 학습 가능
다시 축소: 핵심 정보만 압축

비유: 
책을 읽을 때 → 줄거리(768) 파악
세부 분석 단계 → 모든 세부사항(3072) 고려
최종 정리 → 핵심 결론(768) 도출
```

#### ④ `Block` 클래스 — Transformer 기본 단위

```python
class Block(nn.Module):
    """
    Transformer Block = Attention + MLP + Residual Connection
    
    12개가 쌓여서 Encoder를 구성
    """
    def __init__(self, config):
        super().__init__()
        self.attention_norm = LayerNorm(768, eps=1e-6)  # 정규화
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
    
    def forward(self, x):
        # ① Residual Connection: h = x (복사본 저장)
        h = x
        
        # ② Layer Normalization → Attention
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        
        # ③ Residual 더하기 (입력 + Attention 출력)
        x = x + h  # ← 이것이 Residual Connection!
        # 왜? gradient가 더 잘 흐르도록 → 깊은 네트워크 학습 가능
        
        # ④ 두 번째 Residual
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        return x, weights
```

**Residual Connection 이해:**
```
일반 신경망:    x → Layer → y (layer만 학습)
Residual:       x → Layer → (y + x) (layer + 건너뛰기)

비유: 
"원래 정보 + 새로 배운 정보"를 더하는 것
→ 최악의 경우 layer 출력이 0이면 x만 통과 (안전!)
→ 매우 깊은 네트워크도 안정적으로 학습 가능
```

#### ⑤ `Part_Attention` 클래스 — TransFG의 핵심! ⭐

```python
class Part_Attention(nn.Module):
    """
    Part Selection Module (PSM)
    
    11개 Transformer Block의 Attention을 분석해서
    가장 discriminative한(구분에 중요한) 패치를 선택
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        x: list of attention_weights (11개)
           각 원소: (배치, 12헤드, 785, 785)
        """
        length = len(x)  # = 11
        
        # ① 첫 번째 Attention Map 복사
        last_map = x[0]
        # shape: (B, 12, 785, 785)
        
        # ② 11개의 Attention을 순차적으로 곱함
        #    (행렬곱 = 어텐션을 전파/누적)
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        # 결과: 11개의 Attention이 모두 반영된 최종 Map
        
        # ③ CLS 토큰이 각 패치를 얼마나 주목하는지 추출
        # last_map: (B, 12, 785, 785)
        # [:,:,0,1:] = 모든 배치, 모든 헤드, CLS행, 패치열들
        last_map = last_map[:, :, 0, 1:]
        # shape: (B, 12, 784) — 각 패치의 중요도 점수
        
        # ④ 가장 중요한 패치 인덱스 찾기
        _, max_inx = last_map.max(2)
        # max_inx: (B, 12) — 각 헤드에서 가장 주목한 패치 번호
        
        return _, max_inx
```

**PSM 동작 시각화:**

```
Attention Block 1:  [P1→P2=0.1, P1→P45=0.8, ...]
Attention Block 2:  [P1→P2=0.2, P1→P45=0.6, ...]
...
Attention Block 11: [P1→P2=0.3, P1→P45=0.7, ...]

11개를 행렬곱으로 누적 →
누적 Attention: [P45=0.5(최대!), P23=0.3, P456=0.1, ...]

→ P45가 새의 부리 위치라면:
   모든 레이어에서 "이 패치가 중요하다"고 판단!

max_inx = [45, 23, 456, ...]  (중요 패치 번호들)
```

#### ⑥ `Encoder` 클래스

```python
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList()
        
        # Block 1~11 생성
        for _ in range(12 - 1):  # 11개
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        
        # PSM
        self.part_select = Part_Attention()
        
        # 마지막 Block 12
        self.part_layer = Block(config)
        self.part_norm = LayerNorm(768, eps=1e-6)
    
    def forward(self, hidden_states):
        # ① Block 1~11 통과, attention weights 수집
        attn_weights = []
        for layer in self.layer:  # 11번 반복
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        
        # ② PSM: 중요 패치 선택
        part_num, part_inx = self.part_select(attn_weights)
        part_inx = part_inx + 1  # +1: CLS 토큰이 인덱스 0이므로
        
        # ③ 선택된 패치의 feature 수집
        parts = []
        B, num = part_inx.shape  # B=배치크기, num=선택된 패치 수(12)
        for i in range(B):
            parts.append(hidden_states[i, part_inx[i, :]])
        parts = torch.stack(parts).squeeze(1)
        # shape: (B, 12, 768) — 선택된 12개 패치의 feature
        
        # ④ CLS + 선택된 패치 합치기
        concat = torch.cat(
            (hidden_states[:, 0].unsqueeze(1), parts),
            dim=1
        )
        # shape: (B, 13, 768) — CLS(1) + 선택된 패치(12)
        
        # ⑤ 마지막 Block 12 통과
        part_states, part_weights = self.part_layer(concat)
        part_encoded = self.part_norm(part_states)
        
        return part_encoded
        # shape: (B, 13, 768)
```

#### ⑦ `VisionTransformer` — 최상위 모델

```python
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843,
                 smoothing_value=0, zero_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        
        self.transformer = Transformer(config, img_size)
        
        # 분류 헤드: 768차원 → 클래스 수(200)
        self.part_head = Linear(768, num_classes)
    
    def forward(self, x, labels=None):
        """
        x:      (B, 3, 448, 448) — 배치 이미지
        labels: (B,) — 정답 레이블 (학습 시만 사용)
        """
        # ① Transformer 통과
        part_tokens = self.transformer(x)
        # shape: (B, 13, 768)
        
        # ② CLS 토큰(0번)으로 분류
        part_logits = self.part_head(part_tokens[:, 0])
        # shape: (B, 200) — 각 클래스의 "점수" (로짓)
        
        if labels is not None:
            # ━━ 학습 모드 ━━
            # CrossEntropy Loss 계산
            loss_fct = CrossEntropyLoss()
            part_loss = loss_fct(
                part_logits.view(-1, self.num_classes),
                labels.view(-1)
            )
            
            # Contrastive Loss 계산 (TransFG 추가 손실)
            contrast_loss = con_loss(part_tokens[:, 0], labels.view(-1))
            
            loss = part_loss + contrast_loss
            return loss, part_logits
        else:
            # ━━ 추론 모드 ━━
            return part_logits
```

#### ⑧ `con_loss()` — Contrastive Loss

```python
def con_loss(features, labels):
    """
    Contrastive Loss: 같은 클래스는 가깝게, 다른 클래스는 멀게
    
    예시:
    배치에 황금새 3마리, 파랑새 2마리가 있다면:
    - 황금새끼리: cosine similarity 최대화
    - 황금새 vs 파랑새: cosine similarity 최소화
    """
    B, _ = features.shape  # B = 배치 크기
    
    # ① L2 정규화 (크기를 1로)
    features = F.normalize(features)
    
    # ② Cosine similarity 행렬 계산
    # features: (B, 768)
    cos_matrix = features.mm(features.t())
    # cos_matrix: (B, B) — 각 샘플 쌍의 코사인 유사도
    
    # ③ 같은 클래스 마스크 생성
    pos_label_matrix = torch.stack(
        [labels == labels[i] for i in range(B)]
    ).float()
    # (B, B): 같은 클래스=1, 다른 클래스=0
    neg_label_matrix = 1 - pos_label_matrix
    
    # ④ Positive Loss: 같은 클래스인데 거리가 멀면 페널티
    pos_cos_matrix = 1 - cos_matrix  # 거리 (유사도의 반대)
    
    # ⑤ Negative Loss: 다른 클래스인데 너무 가까우면(>0.4) 페널티
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0  # 0.4 미만은 무시
    
    loss = (pos_cos_matrix * pos_label_matrix).sum() + \
           (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)  # 배치 크기로 정규화
    
    return loss
```

**Contrastive Loss 시각화:**
```
특징 공간 (Feature Space)에서:

학습 전:                    학습 후:
  황금새 ●  파랑새 ●        파랑새●●●     황금새●●●
    ● 파랑새                              (같은 종은 모여!)
  ●황금새 ● 파랑새          ━━━━━━━━━━━━━━━━
                            (다른 종은 멀리!)
```

---

### 6.5 `trainer.py` — 학습 엔진

**역할**: 모델이 데이터를 보고 점점 나아지도록 하는 학습 루프

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 핵심 구조: 학습의 흐름
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ① 스케줄러 (학습률을 조절)
class WarmupCosineSchedule:
    """
    처음에는 천천히 시작 (Warmup)
    그 다음 코사인 곡선으로 서서히 감소
    
    학습률 변화:
    0.03
    │   ╭─────╮
    │  ╭╯      ╰─────────╮
    │ ╭╯                  ╰───────────
    │╭╯                               ╰─
    └─────────────────────────────────▶ step
      ↑ warmup     ↑ cosine decay 시작
    """
```

**학습 루프의 핵심 단계:**

```python
def train(model, train_loader, test_loader, device, ...):
    
    # ━━ 준비 ━━
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, ...)
    #   SGD = 확률적 경사 하강법
    #   gradient(기울기)의 반대 방향으로 가중치 업데이트
    
    scheduler = WarmupCosineSchedule(...)
    #   학습률 스케줄러
    
    scaler = GradScaler()
    #   FP16(반정밀도) 사용 시 gradient 스케일 관리
    
    # ━━ 학습 루프 ━━
    while global_step < num_steps:  # 10000 스텝
        model.train()  # 학습 모드 (Dropout 활성화)
        
        for x, y in train_loader:  # 배치마다
            x, y = x.to(device), y.to(device)  # GPU로 이동
            
            # ① 순전파 (Forward Pass)
            with autocast():  # FP16으로 연산 (속도↑, 메모리↓)
                loss, logits = model(x, y)
                # loss = CrossEntropy Loss + Contrastive Loss
            
            # ② 역전파 (Backward Pass)
            scaler.scale(loss).backward()
            # 각 가중치에 대한 gradient(기울기) 계산
            
            # ③ Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # gradient 크기를 1.0으로 제한
            # 왜? gradient가 너무 크면 학습이 불안정해짐
            
            # ④ 가중치 업데이트
            scaler.step(optimizer)
            # gradient의 반대 방향으로 가중치 이동
            
            # ⑤ 학습률 업데이트
            scheduler.step()
            
            # ⑥ gradient 초기화
            optimizer.zero_grad()
```

**FP16 (Half Precision) 이해:**
```
FP32 (기본):  32비트로 숫자 표현  → 정밀도 높음, 메모리 많이 사용
FP16 (반정밀): 16비트로 숫자 표현  → 약간 덜 정밀, 메모리 절반

GradScaler 역할:
- FP16은 매우 작은 gradient를 표현 못함 (언더플로우)
- 해결: gradient를 큰 수로 곱해서 저장 → 업데이트 직전에 다시 나눔
```

**SGD vs Adam 비교:**
```
SGD:  gradient 방향으로 일정하게 이동
      → 느리지만 안정적, Fine-Grained에 좋음
Adam: gradient + momentum + adaptive learning rate
      → 빠르지만 때로 불안정, 일반 학습에 좋음

TransFG 논문은 SGD를 선택 → Fine-Grained에 더 좋은 성능
```

#### `AverageMeter` 클래스

```python
class AverageMeter:
    """
    평균값을 누적 계산하는 도우미 클래스
    
    사용 예:
    meter = AverageMeter()
    meter.update(0.5)    # loss 0.5
    meter.update(0.3)    # loss 0.3
    print(meter.avg)     # 0.4 (평균)
    print(meter.val)     # 0.3 (가장 최근)
    """
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
```

---

### 6.6 `inference_utils.py` — 추론

**역할**: 학습된 모델로 새로운 이미지를 예측

```python
@torch.no_grad()  # gradient 계산 안 함 (추론에 불필요 + 메모리 절약)
def predict_single(model, image_path, device, class_names=None):
    """
    이미지 파일 하나를 받아서 클래스 예측
    
    returns:
        pred_idx   : 예측된 클래스 번호 (0~199)
        confidence : 해당 클래스의 확률 (0.0~1.0)
        probs      : 모든 클래스의 확률 배열 (200개)
    """
    # 이미지 전처리
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    # .unsqueeze(0): (3, 448, 448) → (1, 3, 448, 448)
    # 배치 차원 추가 (모델은 배치 형태를 기대)
    
    model.eval()  # 평가 모드 (Dropout 비활성화!)
    logits = model(x)           # (1, 200) — 원시 점수
    probs = torch.softmax(logits, dim=-1)[0]  # (200,) — 확률
    pred = int(probs.argmax())  # 가장 높은 확률의 인덱스
    confidence = float(probs[pred])
    
    return {"pred_idx": pred, "confidence": confidence, ...}
```

**Softmax란?**
```
로짓(logits) = 모델이 출력하는 원시 점수
   [-1.2, 3.4, 0.8, -0.5, ...]  (200개)

Softmax = 이것을 확률로 변환 (합 = 1)
   softmax(x_i) = exp(x_i) / Σexp(x_j)

결과:
   [0.01, 0.89, 0.07, 0.02, ...]  ← 합=1, 모두 양수
          ↑
       94% 확률! → 이 클래스가 예측 결과
```

---

### 6.7 `visualization.py` — 시각화

**역할**: 학습 결과를 사람이 이해하기 쉽게 시각화

#### Attention Map 추출 — `AttentionMapExtractor`

```python
class AttentionMapExtractor:
    """
    Forward Hook으로 Attention 가중치 수집
    
    Forward Hook = 모델이 계산하는 도중에 중간 값을 가로채는 기법
    모델 코드를 직접 수정하지 않고도 내부 값 접근 가능!
    """
    
    def __enter__(self):
        # 각 Attention 모듈에 hook 등록
        for layer in self.model.transformer.encoder.layer:
            h = layer.attn.register_forward_hook(self._hook)
            self._hooks.append(h)
    
    def _hook(self, module, input, output):
        # Attention.forward가 반환하는 (attn_output, weights) 중
        # weights를 가로채서 저장
        self.attention_maps.append(output[1].detach().cpu())
    
    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()  # Hook 제거 (안 지우면 메모리 누수!)
```

**Hook 사용 패턴 (Context Manager):**
```python
with AttentionMapExtractor(model) as ext:
    # 이 블록 안에서 model()을 호출하면
    with torch.no_grad():
        _ = model(x)
    # 자동으로 attention maps가 ext.attention_maps에 수집됨
# 블록 밖에서 hooks는 자동으로 제거됨
```

---

## 7. 모델 아키텍처 깊이 파기

### 7.1 Patch Embedding — 이미지를 "단어"처럼

```
입력: 이미지 (3, 448, 448)
    3 = RGB 채널
    448 = 가로/세로 픽셀 수

방법: Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
    kernel_size=16: 16×16 픽셀 영역을 한 번에 처리
    stride=16 (non-overlap): 16픽셀씩 이동 → 겹침 없음
    out_channels=768: 각 패치를 768차원 벡터로 변환

계산:
(448 - 16) / 16 + 1 = 28 → 28×28 = 784개 패치

출력: (배치, 768, 28, 28)
     → flatten, transpose
     → (배치, 784, 768)  ← 784개의 768차원 벡터!
```

**Non-overlap vs Overlap:**
```
Non-overlap (stride=16):          Overlap (stride=12):
┌──┬──┬──┬──┐                   ┌────────┐
│P1│P2│P3│P4│                   │P1      │
├──┼──┼──┼──┤                   │   P2   │
│P5│P6│P7│P8│                   │      P3│
└──┴──┴──┴──┘                   └────────┘
패치가 딱 맞게 분할              패치가 겹침 → 더 많은 패치 → 더 세밀

Non-overlap: 28×28 = 784 patches
Overlap(stride=12): (448-16)/12+1 = 37 → 37×37 = 1369 patches!
→ 더 많은 패치 = 더 풍부한 정보 = 성능 향상 (but 연산 증가)
```

### 7.2 Position Embedding — 순서 정보 추가

```python
self.position_embeddings = nn.Parameter(
    torch.zeros(1, n_patches + 1, 768)
)
# n_patches+1 = 785 (784 패치 + 1 CLS)

# 적용:
embeddings = patch_embeddings + position_embeddings
#             위치 무관 정보   +    위치 정보
```

**왜 필요한가?**
```
패치 임베딩만 있으면:
[P_부리][P_눈][P_날개] 와 [P_날개][P_눈][P_부리]가 같아 보임!
(순서가 달라도 같은 벡터)

위치 임베딩 추가:
[P_부리+위치1][P_눈+위치2][P_날개+위치3]
→ 각 패치가 이미지 어디에 있는지 모델이 알 수 있음!
```

### 7.3 Multi-Head Self-Attention — 가장 중요한 부분

**"Self"의 의미:**
```
Self-Attention: 입력이 자기 자신과 비교
Cross-Attention: 다른 입력과 비교 (번역에서 사용)

우리는 Self-Attention:
입력 [CLS, P1, ..., P784] → 자기들끼리 모든 쌍 비교
→ 785 × 785 = 615,225 쌍의 관계 학습!
```

**"Multi-Head"의 의미:**
```
12개의 Head = 12가지 시각으로 동시에 봄

Head 1: "색깔 패턴에 주목" → 부리 색
Head 2: "형태에 주목"     → 날개 모양
Head 3: "질감에 주목"     → 깃털 패턴
...
Head 12: ...

12개 결과를 합쳐서 768차원 출력
```

**수학적 표현 (무서워 보이지만 사실 간단!):**

```
Q (Query)  = x × W_Q     "질문: 나는 무엇을 찾나?"
K (Key)    = x × W_K     "키: 나는 무엇인가?"
V (Value)  = x × W_V     "값: 내가 줄 정보는?"

Attention(Q,K,V) = softmax(Q × K^T / √d_k) × V
                            ↑유사도계산   ↑확률변환  ↑정보가중합

직관:
- Q × K^T: "질문"과 "키"의 유사도 → 높을수록 주목
- softmax: 합이 1이 되는 확률로 변환
- × V: 높은 Attention weight을 가진 V를 더 많이 사용
```

### 7.4 MLP Block — 정보 처리

```
Attention이 "어디서 정보를 모을까?" 를 결정한다면
MLP는 "모은 정보를 어떻게 처리할까?" 를 담당

[768] → fc1 → [3072] → GELU → fc2 → [768]

GELU 활성화 함수:
- ReLU: max(0, x) (꺾인 형태)
- GELU: x × Φ(x) (더 부드러운 곡선)
- Transformer에서 GELU가 더 좋은 성능
```

### 7.5 Part Selection Module (PSM) — TransFG의 핵심

**아이디어 원리:**

```
일반 ViT:           TransFG:
[CLS, P1...P784]    [CLS, P1...P784]
      ↓                   ↓
[Block 1]           [Block 1] → Attn₁ 저장
[Block 2]           [Block 2] → Attn₂ 저장
...                 ...
[Block 12]          [Block 11] → Attn₁₁ 저장
      ↓                   ↓
  분류기            ▼ PSM: Attn₁×Attn₂×...×Attn₁₁ 누적
                    가장 중요한 12개 패치 선택
                          ↓
                    [CLS, P45, P23, P178, ...]
                    (선택된 패치만으로 구성)
                          ↓
                    [Block 12]
                          ↓
                       분류기
                    
→ 마지막 블록이 "가장 중요한 부위"에만 집중할 수 있음!
```

**Attention 누적곱 이유:**
```
Block1의 Attention: P45가 이미지 전체의 맥락에서 중요
Block2의 Attention: Block1 결과에서 P45가 계속 중요
...
Block11의 Attention: 누적적으로 P45가 가장 중요

곱셈 = 모든 레이어에서 일관되게 중요한 것만 살아남음!
(한 레이어에서만 중요한 것은 곱하면 희석됨)
```

### 7.6 Contrastive Loss — 같은 종끼리 모아라

```
목표: Feature Space에서 같은 종은 가깝게, 다른 종은 멀게

Loss = Positive Loss + Negative Loss

Positive Loss = (1 - cos_sim) × (같은 클래스 마스크)
  → 같은 클래스인데 cosine similarity가 낮으면(멀면) 페널티↑

Negative Loss = max(0, cos_sim - 0.4) × (다른 클래스 마스크)
  → 다른 클래스인데 cosine similarity > 0.4면(가까우면) 페널티↑
  → 0.4보다 멀면 이미 충분히 구분된 것 → 페널티 없음

전체 Loss = CrossEntropy Loss + Contrastive Loss
           (분류 정확도)       (클래스 내 응집도)
```

---

## 8. 학습 전략

### 8.1 전체 학습 흐름

```
PreTrained ViT-B/16 (ImageNet-21k 학습됨)
           ↓
    CUB 데이터로 Fine-Tuning
           ↓
    10,000 steps 학습
           ↓
    Best Validation Accuracy 저장
```

### 8.2 왜 Pretrained를 사용하나?

```
처음부터 학습 (Random Init):
- 수백만 장 이미지, 수주~수개월 필요
- 컴퓨팅 비용 매우 높음

Pretrained + Fine-tuning:
- 이미 이미지의 기본 특징을 학습한 상태에서 시작
- 수시간~수일 만에 좋은 성능 달성
- "어깨에 올라타기" 방식

ImageNet-21k = 21,000가지 분류, 1,400만 장
→ 이미 다양한 시각 특징을 학습함
→ 새 분류에도 응용 가능!
```

### 8.3 Cosine Schedule with Warmup

```
학습률(Learning Rate):
"한 번에 얼마나 크게 가중치를 바꾸나?"

너무 크면: 학습이 불안정, 발산 가능
너무 작으면: 학습이 너무 느림

Warmup: 처음 500 steps는 작게 시작
Cosine: 그 다음 서서히 감소

      3e-2 (0.03)
       ┤         ╭──────────────╮
  lr  ┤        ╭╯              ╰─────────────╮
       ┤      ╭╯                              ╰───
       ┤   ╭╯
       ┼──╯
       0   500                                 10000  step
       ↑↑↑
     warmup
```

### 8.4 Gradient Accumulation

```
실제 배치 크기: 논문 = 4GPU × 16 = 64
우리 환경:     1GPU × 8 = 8

gradient_accumulation_steps = 2 사용 시:
- 8장씩 2번 누적 = 사실상 16장과 같은 효과
- GPU 메모리는 8장 분량만 사용

step 1: batch_a (8장) → gradient 계산, 누적
step 2: batch_b (8장) → gradient 계산, 누적, 업데이트
→ batch_a + batch_b = 16장 효과
```

---

## 9. 노트북 셀별 가이드

### Cell 1 — 환경 확인
```python
import torch
print(torch.cuda.is_available())  # True면 GPU 사용 가능!
```
> **확인 포인트**: GPU가 표시되면 성공. `False`면 TransFG_venv 커널 선택 확인

### Cell 2 — 패키지 설치
이미 TransFG_venv에 설치되어 있으므로 주석 상태. 커널이 맞다면 실행 불필요.

### Cell 3 — CUB 데이터셋 다운로드
> **이미 다운로드 완료!** 실행하면 "이미 존재합니다"라고 뜰 것

```python
# 내부적으로 하는 일:
# 1. urllib.request.urlretrieve(URL, 파일경로) → 다운로드
# 2. tarfile.open(...).extractall(...) → 압축 해제
# 3. 파일 검증 (images.txt, labels.txt 등)
```

### Cell 4 — ViT 가중치 다운로드
> **이미 다운로드 완료!** (412.8 MB)

### Cell 5 — 임포트 및 경로 설정
```python
# sys.path에 "TransFG" 추가하는 이유:
# from models.modeling import ... 가 TransFG/models/modeling.py를 찾아야 함
# Python은 sys.path에 있는 경로들을 검색함

sys.path.insert(0, str(Path("TransFG").resolve()))
# insert(0, ...) = 가장 먼저 검색
```

### Cell 6 — 데이터셋 탐색
```python
# CUBDataset 객체 생성 (이미지는 아직 안 읽음!)
trainset = CUBDataset(root="data/CUB_200_2011", is_train=True)

# len() 호출 → __len__ 실행 → labels 길이 반환
len(trainset)  # 5994

# 인덱싱 → __getitem__ 실행 → 이미지 읽고 반환
img, label = trainset[0]
```

### Cell 7 — 샘플 시각화
```python
# 주의: 정규화된 이미지는 직접 보면 색이 이상함
# denormalize() 함수가 역변환 수행:
# (pixel × std + mean) → 원래 색상으로 복원
```

### Cell 8 — Config 설정

중요 파라미터 설명:
```python
CFG = {
    "split"       : "overlap",   # 패치 겹침 → 성능 더 좋음
    "slide_step"  : 12,          # overlap stride (12픽셀)
    "img_size"    : 448,         # 고해상도 → 세밀한 특징 포착
    "quick_steps" : 200,         # 200 스텝 = 빠른 동작 확인용
    "num_steps"   : 10000,       # 실제 학습은 10000 스텝
    "lr"          : 3e-2,        # 0.03 (SGD에서 꽤 큰 값)
    "fp16"        : True,        # 반정밀도 → 메모리 절약
}
```

### Cell 9 — 데이터로더
```python
# 배치 shape 이해:
# x_batch: torch.Size([8, 3, 448, 448])
#  ↑배치크기  ↑채널(RGB)  ↑높이  ↑너비

# y_batch: torch.Size([8])
#  ↑배치크기 (각각 0~199 사이의 정수)
```

### Cell 10 — 모델 생성

```python
# config.split = 'overlap'로 설정하면:
# Embeddings 클래스에서 stride=slide_step(=12)으로 Conv2d 생성
# → 더 많은 패치 생성 (1369개 vs 784개)

# model.load_from(np.load("pretrained/ViT-B_16.npz"))
# → JAX 가중치를 PyTorch 가중치로 복사
# → "load_pretrained: grid-size from 14 to 28" 메시지가 정상!
#   (원본: 224×224 → 14×14 패치 / 우리: 448×448 → 28×28 패치)
#   → Position Embedding을 보간(interpolation)으로 크기 조정
```

### Cell 11 — 모델 구조 탐색

```python
# Forward pass 테스트:
dummy = torch.randn(2, 3, 448, 448).to(DEVICE)
out = model(dummy)  # labels 없으면 추론 모드
# out: torch.Size([2, 200]) ← 2장, 200 클래스 점수
```

### Cell 12 — 학습 (200 steps)

```
step 1: 손실=2.1 (거의 랜덤)
step 50: 손실=1.8 (조금 나아짐)
step 100: 검증 → 정확도 약 10% (무작위보다 높음: 1/200=0.5%)
step 200: 손실=1.5, 정확도 약 20%
```

> **주의**: 200 steps는 학습이 충분하지 않습니다. 패턴 확인용입니다.  
> 실제 성능(~91%)은 10,000 steps가 필요합니다.

### Cell 14 — 전체 학습 (주석 해제 후)

```
예상 시간: GB10(GB10 단일 GPU)에서 약 8~12시간
팁: 밤에 실행해두고 다음날 확인!

TensorBoard로 실시간 모니터링:
터미널에서: tensorboard --logdir logs
브라우저에서: localhost:6006
```

### Cell 15~16 — 평가

```python
# model.eval() 중요!
# 학습 모드(model.train())와 다른 점:
# - Dropout 비활성화 (매번 같은 결과)
# - BatchNorm이 저장된 통계 사용

# @torch.no_grad() 중요!
# gradient 계산 안 함 → 메모리 절약, 속도 향상
```

### Cell 17 — 단일 이미지 추론

```python
result = predict_single(model, img_path, DEVICE)
# result = {
#     'pred_idx': 45,                     ← 예측 클래스 번호
#     'confidence': 0.8432,               ← 84.3% 확신
#     'pred_name': '046.Gadwall',          ← 클래스 이름
#     'top5': [('046.Gadwall', 0.84), ...] ← 상위 5개
# }
```

### Cell 19~20 — Attention 시각화

```
출력 이미지 해석:
- 빨간색 = 높은 Attention (모델이 주목하는 곳)
- 파란색 = 낮은 Attention (별로 안 봄)

초기 레이어(Layer 1~3): 주로 전역 패턴
후기 레이어(Layer 9~11): 새의 특정 부위 (부리, 눈 등)
→ 레이어가 깊어질수록 더 구체적인 특징에 집중
```

---

## 10. 자주 묻는 질문 (FAQ)

**Q: CUDA warning이 계속 뜨는데 괜찮나요?**
```
"Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)"

→ GB10은 12.1, PyTorch는 최대 12.0 지원 → 경고 발생
→ 하지만 하위 호환성으로 정상 동작합니다. 무시해도 됩니다.
```

**Q: `scipy.misc.imread`가 뭔지 왜 오류가 나나요?**
```
scipy.misc.imread = scipy 라이브러리의 이미지 읽기 함수
하지만 scipy 1.3.0 이후 제거됨 (2019년)

TransFG 원본 코드(2021년)가 이미 제거된 함수를 사용
→ dataset_cub_fixed.py에서 PIL.Image.open으로 교체했습니다
```

**Q: overlap과 non-overlap 차이가 성능에 얼마나 영향을 주나요?**
```
논문 결과 (CUB-200-2011):
- Non-overlap: 90.8%
- Overlap (slide_step=12): 91.7%

약 0.9% 차이 → Fine-Grained에서 의미 있는 차이!
대신 연산량이 약 1.7배 증가 (784개 → 1369개 패치)
```

**Q: 학습이 너무 오래 걸리는데 어떻게 하나요?**
```
방법 1: quick_steps=200으로 동작 확인 후 overnight 학습
방법 2: img_size=224로 줄이기 (빠르지만 성능 저하)
방법 3: train_batch를 줄이기 (8→4, 더 자주 업데이트)
방법 4: eval_every를 늘리기 (100→500, 평가 빈도 감소)
```

**Q: loss가 줄어들지 않아요**
```
체크리스트:
1. model.train()이 학습 루프 안에 있는지 확인
2. optimizer.zero_grad()가 매 step 실행되는지 확인
3. loss.backward() 호출하는지 확인
4. learning_rate가 너무 크거나 작지 않은지 확인
5. 데이터가 올바르게 로드되는지 (x.shape, y.shape) 확인
```

**Q: 모델 저장/로드는 어떻게 하나요?**
```python
# 저장 (trainer.py에서 자동으로 수행)
torch.save({'model': model.state_dict()}, 'output/checkpoint.bin')

# 로드 (Cell 15에서)
ckpt = torch.load('output/checkpoint.bin', map_location=device)
model.load_state_dict(ckpt['model'])
```

---

## 11. 핵심 용어 사전

| 용어 | 설명 |
|------|------|
| **Tensor** | PyTorch의 다차원 배열. NumPy와 비슷하지만 GPU에서 동작 |
| **Batch** | 여러 샘플을 묶어 한 번에 처리하는 단위 |
| **Epoch** | 전체 학습 데이터를 한 번 다 본 것 |
| **Loss** | 모델의 예측이 얼마나 틀렸는지 수치화한 것 (낮을수록 좋음) |
| **Gradient** | 각 가중치에 대한 loss의 변화율 → 업데이트 방향 |
| **Backpropagation** | loss에서 거꾸로 계산해 gradient 구하는 과정 |
| **Optimizer** | gradient를 이용해 가중치를 업데이트하는 알고리즘 |
| **Learning Rate (lr)** | 한 번에 얼마나 크게 가중치를 바꾸나 |
| **Overfitting** | 학습 데이터에만 잘 맞고 새 데이터에는 못하는 현상 |
| **Dropout** | 학습 시 일부 뉴런을 무작위로 끄는 정규화 기법 |
| **LayerNorm** | 레이어별 정규화 → 학습 안정화 |
| **Fine-tuning** | 사전학습된 모델을 특정 태스크에 맞게 추가 학습 |
| **Attention** | 중요한 부분에 더 집중하도록 가중치를 학습 |
| **CLS token** | 분류에 사용되는 특별 토큰, 전체 이미지를 대표 |
| **Logit** | Softmax 전의 원시 점수 (양수/음수 가능) |
| **Softmax** | 로짓을 0~1 사이 확률로 변환 (합=1) |
| **Top-1 Accuracy** | 1위 예측이 맞는 비율 |
| **Top-5 Accuracy** | 상위 5개 예측 중 정답이 있는 비율 |
| **Pretrained** | 다른 큰 데이터셋으로 미리 학습된 가중치 |
| **FP16 / AMP** | 16비트 부동소수점, 메모리 절약 + 속도 향상 |
| **GradScaler** | FP16 학습에서 gradient 안정화 도구 |
| **DataLoader** | 데이터를 배치로 자동 제공하는 PyTorch 도구 |
| **Hook** | 모델의 중간 값을 가로채는 PyTorch 메커니즘 |
| **Cosine Similarity** | 두 벡터의 각도로 유사도 측정 (-1~1) |

---

## 📋 학습 체크리스트

### 기본 이해 ✓
- [ ] Fine-Grained Recognition이 일반 분류와 다른 이유를 말할 수 있다
- [ ] Attention 메커니즘이 무엇인지 자신의 말로 설명할 수 있다
- [ ] Patch Embedding이 이미지를 어떻게 변환하는지 안다
- [ ] CLS 토큰의 역할을 안다

### 코드 이해 ✓
- [ ] CUBDataset이 어떻게 이미지를 읽는지 안다 (`__getitem__`)
- [ ] DataLoader가 배치를 만드는 과정을 안다
- [ ] 학습 루프의 4단계 (forward→backward→clip→step)를 안다
- [ ] model.eval() vs model.train()의 차이를 안다

### TransFG 이해 ✓
- [ ] PSM (Part Selection Module)이 하는 일을 설명할 수 있다
- [ ] Contrastive Loss가 일반 CrossEntropy와 다른 점을 안다
- [ ] Overlap Patch가 Non-overlap보다 좋은 이유를 안다

### 실험 ✓
- [ ] Cell 1~9를 에러 없이 실행했다
- [ ] Cell 12 (200 steps)를 실행했다
- [ ] Attention Map 시각화를 확인했다
- [ ] 직접 이미지 하나를 추론했다

---

## 🚀 더 나아가기 — 실험 아이디어

1. **모델 크기 변경**: `ViT-B_32` (패치 크기 32×32)로 바꿔보기
2. **이미지 해상도**: `img_size=224`로 줄여서 속도 비교
3. **Split 방식**: `non-overlap` vs `overlap` 성능 비교
4. **학습률**: `lr=1e-2` vs `lr=3e-2` 비교
5. **Label Smoothing**: `smoothing=0.1` 추가 효과 확인

---

*이 가이드는 TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI 2022) 논문을 기반으로 작성되었습니다.*
