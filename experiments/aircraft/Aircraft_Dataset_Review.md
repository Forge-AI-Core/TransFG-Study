# FGVC-Aircraft Dataset Review

> **Fine-Grained Visual Categorization of Aircraft (FGVC-Aircraft)**
> 항공기 모델 세밀 분류를 위한 벤치마크 데이터셋

---

## 1. 개요

FGVC-Aircraft는 Oxford Visual Geometry Group(VGG)에서 2013년에 공개한 **세밀 이미지 분류(Fine-Grained Visual Categorization, FGVC)** 분야의 대표적 벤치마크 데이터셋이다. 약 100종의 항공기 모델(variant)에 해당하는 10,200장의 이미지로 구성되어 있으며, **계층적 레이블링** (variant → family → manufacturer)이 특징이다.

| 항목 | 내용 |
|------|------|
| 발표 연도 | 2013 |
| 제작 기관 | Oxford Visual Geometry Group (VGG) |
| 전체 이미지 수 | 10,200장 |
| 클래스 수 (variants) | **100종** |
| 클래스 수 (families) | 70종 |
| 클래스 수 (manufacturers) | 30종 |
| 클래스당 이미지 수 | 100장 (균일) |
| 어노테이션 종류 | 클래스 레이블 (3단계), 바운딩 박스 |
| 라이선스 | 연구 목적 비상업적 사용 |
| 공식 페이지 | https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |
| 인용 | Maji et al., arXiv:1306.5151 (2013) |

---

## 2. 디렉토리 구조

```
fgvc-aircraft-2013b/
├── data/
│   ├── images/                                # 10,200장 (전체 .jpg)
│   │   ├── 0034309.jpg
│   │   ├── 0034958.jpg
│   │   └── ... (7-digit ID)
│   │
│   ├── images_variant_train.txt               # 학습셋 variant 레이블 (3,334행)
│   ├── images_variant_val.txt                 # 검증셋 variant 레이블 (3,333행)
│   ├── images_variant_test.txt                # 테스트셋 variant 레이블 (3,333행)
│   ├── images_variant_trainval.txt            # train+val 통합 (6,667행)
│   │
│   ├── images_family_{train,val,test}.txt     # family-level 레이블 (70 classes)
│   ├── images_manufacturer_{train,val,test}.txt # manufacturer-level (30 classes)
│   │
│   ├── variants.txt                           # 100 variants 목록
│   ├── families.txt                           # 70 families 목록
│   ├── manufacturers.txt                      # 30 manufacturers 목록
│   │
│   ├── images_box.txt                         # 바운딩 박스 (10,200행)
│   └── images.txt                             # 전체 이미지 ID 목록
│
├── evaluation.m                               # MATLAB 평가 스크립트
└── README.html                                # 공식 데이터셋 설명
```

### 레이블 파일 형식

```
# images_variant_train.txt 예시
# 형식: <image_id> <variant_name>
1234567 Boeing 707-320
2345678 Airbus A320
...
```

---

## 3. 클래스 계층 구조

FGVC-Aircraft의 특징은 **3단계 계층적 레이블링**이다:

```
Manufacturer (30종)       Family (70종)         Variant (100종)
─────────────             ──────────            ───────────
Boeing             ─→     Boeing 707     ─→     Boeing 707-320
                          Boeing 737     ─→     Boeing 737-300
                                         ─→     Boeing 737-400
                          Boeing 747     ─→     Boeing 747-100
                                         ─→     Boeing 747-200
                                         ─→     Boeing 747-400
Airbus             ─→     A320 family    ─→     Airbus A318
                                         ─→     Airbus A319
                                         ─→     Airbus A320
                                         ─→     Airbus A321
McDonnell Douglas  ─→     DC-9           ─→     DC-9-30
                                         ─→     DC-9-40
```

### 대표 제조사 (Manufacturer)

| 제조사 | 비고 |
|---|---|
| Boeing | 707, 727, 737, 747, 757, 767, 777 등 |
| Airbus | A300, A310, A318, A319, A320, A321, A330, A340, A380 |
| McDonnell Douglas | DC-3, DC-6, DC-8, DC-9, DC-10, MD-11, MD-80/90 |
| Lockheed | C-130, L-1011, SR-71, P-3 등 |
| Embraer | ERJ-135, ERJ-145, EMB-120 등 |
| Bombardier | CRJ 시리즈, Challenger, Global Express |
| Cessna | 172, 208 (Caravan), Citation 시리즈 |
| Tupolev | Tu-134, Tu-154 등 |

### 본 실험에서 사용하는 레이블

**Variant 레벨(100 classes)** — `project_config.py`의 `NUM_CLASSES = 100`

---

## 4. 학습/검증/테스트 분할

| 분할 | 이미지 수 | 비율 | 파일 |
|------|---------|------|---|
| Train | 3,334 | 32.7% | `images_variant_train.txt` |
| Val | 3,333 | 32.7% | `images_variant_val.txt` |
| Test | 3,333 | 32.7% | `images_variant_test.txt` |
| **Total** | **10,000** | **98.0%** | (200장은 추가 이미지) |

> 💡 **CUB와 비교**: CUB는 50:50 train/test, Aircraft는 33:33:33 train/val/test
> 클래스당 균일하게 100장 → 클래스 불균형 문제 없음

---

## 5. 어노테이션 상세

### 5-1. 바운딩 박스 (Bounding Box)

모든 이미지에 항공기 영역을 감싸는 단일 바운딩 박스 제공:

```
# images_box.txt 형식: <image_id> <xmin> <ymin> <xmax> <ymax>  (픽셀 단위)
0034309 130 32 1481 538
0034958  78 24 1029 432
```

> 학습 시 crop으로 활용 가능 (선택적). 본 실험은 전체 이미지 사용.

### 5-2. 워터마크/저작권 영역 처리

원본 이미지 하단에 **20-pixel 높이의 저작권 텍스트**가 있어, 일반적으로 **하단 20px 잘라내고 사용**한다. 본 실험의 `dataset_aircraft.py`에서 자동 처리.

```python
# dataset_aircraft.py 내부에서
image = image.crop((0, 0, width, height - 20))  # 저작권 영역 제거
```

---

## 6. 샘플 이미지

> 💡 데이터셋 다운로드 후 `data/images/` 폴더의 이미지 사용

### Boeing 747-400
> 점보제트의 대명사. 4발 와이드바디, 2층 구조 특유의 둥근 코쿠핏.

### Airbus A380
> 세계 최대 여객기. 2층 풀더블데크, 4발 엔진.

### McDonnell Douglas DC-3
> 1930년대 클래식 프로펠러 항공기. 항공사 시대의 시작.

### Lockheed SR-71 (Blackbird)
> 최고 속도 마하 3+ 정찰기. 검은 도색과 날렵한 디자인.

### Concorde
> 초음속 여객기 (Aérospatiale/BAC 합작). 삼각날개와 가변 노즈.

> 📷 실제 샘플 이미지를 README에 추가하려면, 데이터 다운로드 후 본 문서에 이미지 임베드 가능.

---

## 7. FGVC 분야에서의 난이도

FGVC-Aircraft가 어려운 이유:

1. **세대 간 미세한 차이**: Boeing 737-300 vs 737-400 vs 737-500은 길이/엔진 외에 거의 동일
2. **자세/각도 다양성**: 측면, 정면, 이착륙 등 다양한 시점
3. **배경 변화**: 공항 활주로, 격납고, 공중, 비행장 등
4. **유사 패밀리 간 구분**: A320 family (A318/A319/A320/A321)는 동체 길이만 다름
5. **워터마크 처리**: 하단 20px 영역 자동 제거 필요

### 주요 벤치마크 성능 (Top-1 Accuracy, Variant 레벨)

| 모델 | 정확도 | 비고 |
|------|--------|------|
| ResNet-50 (ImageNet pretrained) | ~84% | 일반 분류 기반 |
| ViT-B/16 (ImageNet pretrained) | ~89% | 비전 트랜스포머 |
| **TransFG (ImageNet pretrained)** | **~92.5%** | 원논문 결과 |
| **TransFG (CUB pretrained, 본 실험 가설)** | **TBD** | Phase 1 검증 대상 |
| SOTA (2023~) | ~95%+ | 다양한 어텐션 기법 |

---

## 8. TransFG와의 연관성

TransFG는 ViT의 어텐션 맵에서 판별력 높은 패치를 선택하는 **Part Selection Module**을 도입.
FGVC-Aircraft에서의 활용:

- **입력**: 448×448 리사이즈 후 ViT patch embedding
- **핵심 아이디어**: 다양한 비행 자세에서도 항공기 특징적 부위(엔진, 노즈, 날개)에 어텐션 집중
- **바운딩 박스**: 학습 시 crop 활용 가능 (선택적)
- **저작권 영역**: 하단 20px 자동 제거하여 ViT 입력 크기 보정

```python
# project_config.py에서의 Aircraft 설정
DATASET     = 'FGVC-Aircraft'
NUM_CLASSES = 100
IMG_SIZE    = 448
PRETRAINED_WEIGHTS = CUB_PRETRAINED  # Phase 1 핵심
```

---

## 9. CUB vs Aircraft 비교

| 항목 | CUB-200-2011 | FGVC-Aircraft |
|---|---|---|
| 도메인 | 자연 (조류) | 인공물 (항공기) |
| 클래스 수 | 200 | 100 |
| 이미지 수 | 11,788 | 10,200 |
| 학습/테스트 | 50:50 | 33:33:33 (train:val:test) |
| 클래스 균형 | 평균 약 58장 | 100장 (균일) |
| 부가 어노테이션 | bbox + 15부위 + 312속성 | bbox + 계층 레이블 |
| 시점 다양성 | 보통 (조류 자세) | **높음** (이착륙, 측면, 공중) |
| 클래스 간 유사도 | 동일 속(genus) | **세대(737-300/-400/-500)** |

### Phase 1 가설: CUB→Aircraft 전이학습 가능한가?

| 가설 항목 | 근거 |
|---|---|
| ✅ 둘 다 fine-grained 분류 | 미세한 시각적 차이를 학습한 표현이 유용 |
| ✅ Part-based attention 학습 | 조류 부위 → 항공기 부위(엔진/날개)로 전이 가능 추정 |
| ⚠️ 도메인 갭 | 자연 vs 인공물 — 저수준 특징은 다를 수 있음 |
| ⚠️ 클래스 수 변화 | 200 → 100, classifier head 재학습 필요 |

> 본 실험에서 이 가설을 검증한다 (ImageNet baseline 대비 CUB pretrained 성능 비교).

---

## 10. 인용 정보

```bibtex
@techreport{maji2013fine,
  title={Fine-grained visual classification of aircraft},
  author={Maji, Subhransu and Rahtu, Esa and Kannala, Juho and Blaschko, Matthew and Vedaldi, Andrea},
  institution={arXiv:1306.5151},
  year={2013}
}
```

---

## 11. References

- 공식 페이지: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- 다운로드: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
- 논문: [Maji et al., 2013 (arXiv:1306.5151)](https://arxiv.org/abs/1306.5151)
- TransFG 원논문: [He et al., 2021 (arXiv:2103.07976)](https://arxiv.org/abs/2103.07976)
