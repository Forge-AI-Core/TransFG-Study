# Stanford Cars Dataset Review

> **Fine-Grained Visual Categorization of Cars (Stanford Cars)**
> 자동차 모델 세밀 분류를 위한 벤치마크 데이터셋

---

## 1. 개요

Stanford Cars는 Stanford AI Lab의 Krause 등이 2013년에 공개한 **세밀 이미지 분류(Fine-Grained Visual Categorization, FGVC)** 분야의 대표적 벤치마크 데이터셋이다. 약 196종의 자동차 모델(make/model/year 조합)에 해당하는 16,185장의 이미지로 구성되어 있으며, **제조사·모델·연식**의 조합으로 클래스가 정의된다.

| 항목 | 내용 |
|------|------|
| 발표 연도 | 2013 |
| 제작 기관 | Stanford AI Lab (Krause et al.) |
| 전체 이미지 수 | 16,185장 |
| 클래스 수 | **196종** (Make + Model + Year) |
| Train | 8,144장 (≈클래스당 41.5장) |
| Test | 8,041장 (≈클래스당 41.0장) |
| 어노테이션 종류 | 클래스 레이블 (이름), 바운딩 박스 |
| 라이선스 | 연구 목적 비상업적 사용 |
| 공식 페이지 (구) | https://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| HuggingFace 미러 | https://huggingface.co/datasets/tanganke/stanford_cars |
| 인용 | Krause et al., 3dRR-13 (2013) |

> ⚠️ **공식 페이지 접속 불가**: 2024년 이후 Stanford의 원본 다운로드 링크가 불안정. 본 실험은 **HuggingFace 미러 (`tanganke/stanford_cars`)** 의 parquet 형식을 사용한다.

---

## 2. 디렉토리 구조 (HuggingFace parquet 형식)

```
Stanford_Cars/
├── README.md                              # 클래스 ID ↔ 이름 매핑 포함
└── data/
    ├── train-00000-of-00001.parquet       # 학습셋 (8,144장 이미지 + 레이블)
    └── test-00000-of-00001.parquet        # 테스트셋 (8,041장 이미지 + 레이블)
```

### Parquet 파일 내부 스키마

```
column 'image':  dict { 'bytes': <원본 JPEG bytes>, 'path': <원본 파일명> }
column 'label':  int  (0 ~ 195)
```

이미지가 **bytes로 parquet 안에 내장**되어 있어 별도 이미지 폴더가 필요 없다. `dataset_stanford_cars.py`가 다음 흐름으로 처리:

```python
img_bytes = parquet_row["image"]["bytes"]    # bytes 추출
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # PIL로 디코딩
```

### 클래스 이름 (README.md 메타데이터)

HuggingFace의 README.md 안에 클래스 ID ↔ 이름 매핑이 yaml 풍 텍스트로 포함:

```
'0': 'AM General Hummer SUV 2000',
'1': 'Acura RL Sedan 2012',
'2': 'Acura TL Sedan 2012',
...
'195': 'smart fortwo Convertible 2012',
```

`dataset_stanford_cars._parse_class_names()`이 정규식으로 추출한다.

---

## 3. 클래스 정의 — Make + Model + Year

Stanford Cars의 특징은 **제조사 + 모델 + 연식 조합**으로 196 클래스를 만든 것:

```
Class 0  : AM General Hummer SUV 2000
Class 1  : Acura RL Sedan 2012
Class 2  : Acura TL Sedan 2012
Class 3  : Acura TL Type-S 2008
Class 4  : Acura TSX Sedan 2012
...
Class 28 : Audi A5 Coupe 2012
Class 29 : Audi R8 Coupe 2012
Class 30 : Audi RS 4 Convertible 2008
...
Class 195: smart fortwo Convertible 2012
```

### 대표 제조사

| 제조사 | 대표 클래스 |
|---|---|
| **GM 계열** (Chevrolet, GMC, Cadillac, Buick, Hummer, Pontiac) | Chevrolet Corvette, Camaro, Tahoe, Cadillac CTS-V 등 |
| **Ford** | Ford Mustang, F-150, Focus, GT 등 |
| **BMW** | BMW M3 Coupe, X5 SUV, 1·3·6 Series 등 |
| **Audi** | A5, R8, S4, RS4, TT 등 |
| **Mercedes-Benz** | C-Class, E-Class, S-Class, SL-Class, SLS AMG 등 |
| **Toyota** | Camry, Corolla, 4Runner SUV, Sequoia SUV |
| **Honda** | Accord, Civic, Odyssey, Pilot |
| **Aston Martin** | V8 Vantage, Virage Coupe, DB9 |
| **Ferrari** | 458 Italia, FF Coupe, California |
| **Lamborghini** | Aventador, Gallardo, Diablo |
| **Porsche** | 911, Boxster, Cayenne |
| **smart** | fortwo Convertible |

### 본 실험에서 사용하는 레이블

**Single-level classification (196 classes)** — `project_config.py`의 `NUM_CLASSES = 196`

Aircraft처럼 계층(variant/family/manufacturer) 구분은 없음.

---

## 4. 학습/테스트 분할

| 분할 | 이미지 수 | 비율 | 파일 |
|------|---------|------|---|
| Train | 8,144 | 50.3% | `train-00000-of-00001.parquet` |
| Test | 8,041 | 49.7% | `test-00000-of-00001.parquet` |
| **Total** | **16,185** | **100%** | |

> 💡 **CUB / Aircraft와 비교**:
> - CUB: 50:50 train/test
> - Aircraft: 33:33:33 train/val/test (3분할)
> - **Cars: 50:50 train/test (val 없음)** ← 본 실험에선 test를 평가에 사용

---

## 5. 어노테이션 상세

### 5-1. 바운딩 박스 (원본만)

원본 Stanford Cars는 자동차 영역을 감싸는 단일 바운딩 박스를 제공:

```
x1, y1, x2, y2 (픽셀 단위)
```

**다만** HuggingFace `tanganke/stanford_cars` 미러는 **이미지 + 레이블만 제공**하며 바운딩 박스는 포함하지 않는다. 본 실험은 전체 이미지를 사용하므로 영향 없음.

### 5-2. 워터마크/저작권 영역

Aircraft와 달리 Stanford Cars는 **워터마크 없음** — 별도 crop 처리 불필요.

```python
# dataset_stanford_cars.py — 단순 디코딩만
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
if self.transform is not None:
    img = self.transform(img)
return img, self.labels[index]
```

---

## 6. 샘플 이미지 (대표 클래스)

> 💡 데이터셋 다운로드 후 `Stanford_Cars/data/train-*.parquet` 의 이미지를 디코딩하여 확인 가능

### 1. AM General Hummer SUV 2000 (Class 0)
> 미군 험비의 민간 버전. 각진 바디, 와이드 휠.

### 2. BMW M3 Coupe 2012
> 독일 스포츠 쿠페의 대명사. 키드니 그릴, 듀얼 머플러.

### 3. Bugatti Veyron 16.4 Coupe 2009
> 1,000+ 마력 하이퍼카. 곡선 위주 디자인.

### 4. Ford F-150 Regular Cab 2012
> 미국 픽업트럭 대표. 대형 그릴, 적재함.

### 5. Tesla Model S Sedan 2012
> 초기 전기차 세단. 매끈한 표면, 그릴 없음.

### 6. smart fortwo Convertible 2012 (Class 195)
> 초소형 2인승 컨버터블. 짧은 전장.

> 📷 실제 샘플 이미지를 README에 추가하려면, parquet 데이터 다운로드 후 jupyter notebook의 `show_sample_grid()` 호출.

---

## 7. FGVC 분야에서의 난이도

Stanford Cars가 어려운 이유:

1. **세대 간 미세한 차이**: 같은 모델의 연식별 변화(예: BMW 3 Series 2007 vs 2012)는 헤드라이트·범퍼 디자인만 다름
2. **모델 그룹 간 유사도**: 동일 제조사의 다른 모델(예: Audi A4 / A5 / A6)이 유사한 디자인 언어 공유
3. **자세/각도 다양성**: 광고용 측면샷, 정면, ¾각도, 후방 등 모든 시점 등장
4. **배경 변화**: 쇼룸, 도로, 자연 배경 등 다양
5. **세련된 디자인 vs 노후 모델**: 2000년대 초반 모델과 2012년 모델이 같은 분류 대상

### 주요 벤치마크 성능 (Top-1 Accuracy)

| 모델 | 정확도 | 비고 |
|------|--------|------|
| ResNet-50 (ImageNet pretrained) | ~91% | 일반 분류 기반 |
| ViT-B/16 (ImageNet pretrained) | ~93% | 비전 트랜스포머 |
| **TransFG (ImageNet pretrained)** | **~94.8%** | 원논문 결과 |
| **TransFG (Aircraft pretrained, 본 실험 가설)** | **TBD** | Phase 2 검증 대상 |
| SOTA (2023~) | ~96.1%+ | 다양한 어텐션 기법 |

---

## 8. TransFG와의 연관성

TransFG는 ViT의 어텐션 맵에서 판별력 높은 패치를 선택하는 **Part Selection Module (PSM)** 을 도입.
Stanford Cars에서의 활용:

- **입력**: 448×448 리사이즈 후 ViT patch embedding
- **핵심 아이디어**: 자동차 식별 부위(헤드라이트, 그릴, 휠, 사이드 라인)에 어텐션 집중
- **고품질 이미지**: 광고용 이미지가 많아 부위별 디테일이 선명함 → PSM이 효과적으로 작동 가능
- **워터마크 처리 불필요**: Aircraft와 달리 별도 crop 없음

```python
# project_config.py에서의 Stanford Cars 설정
DATASET     = 'Stanford_Cars'
NUM_CLASSES = 196
IMG_SIZE    = 448
PRETRAINED_WEIGHTS = AIRCRAFT_PRETRAINED  # Phase 2 핵심
```

---

## 9. CUB vs Aircraft vs Stanford Cars 비교

| 항목 | CUB-200-2011 | FGVC-Aircraft | **Stanford Cars** |
|---|---|---|---|
| 도메인 | 자연 (조류) | 인공물 (항공기) | **인공물 (자동차)** |
| 클래스 수 | 200 | 100 | **196** |
| 이미지 수 | 11,788 | 10,200 | **16,185** |
| 학습/테스트 | 50:50 | 33:33:33 | **50:50 (val 없음)** |
| 클래스당 평균 | 약 58장 | 100장 (균일) | 약 41~42장 |
| 부가 어노테이션 | bbox + 15부위 + 312속성 | bbox + 계층 레이블 | bbox (원본만) |
| 데이터 배포 형식 | 이미지 폴더 + txt | 이미지 폴더 + txt | **parquet (bytes 내장)** |
| 시점 다양성 | 보통 | 높음 (이착륙, 측면) | **매우 높음** (광고샷 다양) |
| 클래스 간 유사도 | 동일 속(genus) | 동일 family (737-300/-400) | **동일 모델 연식 변형** |

### Phase 2 가설: Aircraft→Cars 전이학습 가능한가?

| 가설 항목 | 근거 |
|---|---|
| ✅ 둘 다 fine-grained 분류 | 미세한 시각적 차이를 학습한 표현 유용 |
| ✅ 둘 다 인공물·금속·직선 | 저수준 특징(엣지, 텍스처) 공유 |
| ✅ Part-based attention 학습 | 항공기 동체/날개/꼬리 → 자동차 차체/헤드라이트/그릴 전이 추정 |
| ⚠️ 클래스 수 변화 | 100 → 196, classifier head 재학습 필요 (`part_head` 제외 로드) |
| ⚠️ 절대 정확도 손해 | Aircraft 78.97% < CUB 90.84%, "정확도 vs 유사도" 트레이드오프 |

> 본 실험에서 이 가설을 검증한다 (ImageNet baseline 1.37% 대비 Aircraft pretrained 향상폭 비교).

---

## 10. 인용 정보

```bibtex
@inproceedings{krause2013collecting,
  title={3D Object Representations for Fine-Grained Categorization},
  author={Krause, Jonathan and Stark, Michael and Deng, Jia and Fei-Fei, Li},
  booktitle={4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13)},
  year={2013}
}
```

---

## 11. References

- 공식 페이지 (구): https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- HuggingFace 미러: https://huggingface.co/datasets/tanganke/stanford_cars
- 논문 PDF: https://ai.stanford.edu/~jkrause/papers/3drr13.pdf
- TransFG 원논문: [He et al., 2021 (arXiv:2103.07976)](https://arxiv.org/abs/2103.07976)
