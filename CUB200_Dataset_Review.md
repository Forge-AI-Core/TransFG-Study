# CUB-200-2011 Dataset Review

> **Caltech-UCSD Birds-200-2011 (CUB-200-2011)**
> 세밀한 조류 이미지 분류를 위한 벤치마크 데이터셋

---

## 1. 개요

CUB-200-2011은 Caltech와 UCSD가 공동으로 구축한 **세밀 이미지 분류(Fine-Grained Visual Categorization, FGVC)** 분야의 대표적인 벤치마크 데이터셋이다. 200종의 북미 조류 사진 약 11,788장으로 구성되어 있으며, 각 이미지에 클래스 레이블, 바운딩 박스, 15개 부위(part) 위치, 312개 이진 속성(attribute) 레이블이 함께 제공된다.

| 항목 | 내용 |
|------|------|
| 발표 연도 | 2011 |
| 클래스 수 | 200종 (북미 조류) |
| 전체 이미지 수 | 11,788장 |
| 학습 이미지 수 | 5,994장 |
| 테스트 이미지 수 | 5,794장 |
| 클래스당 이미지 수 | 최소 41장 / 최대 60장 / 평균 58.9장 |
| 어노테이션 종류 | 클래스 레이블, 바운딩 박스, 15개 부위 위치, 312개 속성 |
| 라이선스 | 연구 목적 비상업적 사용 |

---

## 2. 디렉토리 구조

```
CUB_200_2011/
├── images/                         # 200개 클래스 폴더 (각 40~60장)
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   ├── ...
│   └── 200.Common_Yellowthroat/
├── images.txt                      # 이미지 ID → 파일 경로 매핑 (11,788행)
├── classes.txt                     # 클래스 ID → 클래스 이름 (200행)
├── image_class_labels.txt          # 이미지 ID → 클래스 ID
├── train_test_split.txt            # 이미지 ID → 학습(1)/테스트(0) 구분
├── bounding_boxes.txt              # 이미지 ID → (x, y, width, height)
├── attributes.txt                  # 속성 ID → 속성 이름 (312행)
├── attributes/
│   ├── image_attribute_labels.txt  # 이미지별 속성 레이블 (MTurk)
│   ├── class_attribute_labels_continuous.txt  # 클래스별 연속 속성값
│   └── certainties.txt             # 응답 확신도 레이블
└── parts/
    ├── parts.txt                   # 부위 ID → 부위 이름 (15개)
    ├── part_locs.txt               # 이미지별 부위 위치 (176,820행)
    └── part_click_locs.txt         # MTurk 다중 부위 클릭 위치
```

---

## 3. 클래스 구성

총 200종의 조류로 구성되며, 알파벳 순서로 번호가 부여된다.

| 범위 | 대표 종 |
|------|---------|
| 001~010 | Black-footed Albatross, Laysan Albatross, Sooty Albatross, Groove-billed Ani, Auklet 4종... |
| 011~030 | Blackbird 4종, Bobolink, Bunting 3종, Cardinal, Catbird 2종... |
| 050~070 | Crow, Cuckoo, Flicker, Flycatcher, Gnatcatcher... |
| 100~120 | Kingfisher, Kingbird, Kittiwake, Lark... |
| 150~170 | Sparrow, Starling, Tern, Thrasher... |
| 180~200 | Warbler, Waterthrush, Waxwing, Woodpecker, Wren, Yellowthroat |

**마지막 10종 예시:**
```
191. Red_headed_Woodpecker
192. Downy_Woodpecker
193. Bewick_Wren
194. Cactus_Wren
195. Carolina_Wren
196. House_Wren
197. Marsh_Wren
198. Rock_Wren
199. Winter_Wren
200. Common_Yellowthroat
```

---

## 4. 샘플 이미지

### 001. Black-footed Albatross (검은발 신천옹)
> 바다 위를 비행하는 대형 해조류. 날개폭이 넓고 몸 전체가 어두운 갈색~흑색.

![Black-footed Albatross](./CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg)

---

### 016. Painted Bunting (칠면조 멧새)
> 수컷의 경우 파랑(머리)·빨강(배)·초록(등)의 화려한 삼색 깃털이 특징.

![Painted Bunting](./CUB_200_2011/images/016.Painted_Bunting/Painted_Bunting_0001_16585.jpg)

---

### 013. Bobolink (보볼링크)
> 번식기 수컷은 검은 몸에 흰 어깨·노란색 뒷머리가 특징인 북미 들판 조류.

![Bobolink](./CUB_200_2011/images/013.Bobolink/Bobolink_0001_9261.jpg)

---

### 088. Western Meadowlark (서부 초원종다리)
> 밝은 노란 배에 검은 V자 가슴 무늬가 특징. 아름다운 울음소리로 유명.

![Western Meadowlark](./CUB_200_2011/images/088.Western_Meadowlark/Western_Meadowlark_0001_78676.jpg)

---

### 192. Downy Woodpecker (솜털 딱따구리)
> 흰·검 체크 패턴에 수컷은 붉은 뒷머리 반점. 북미에서 가장 작은 딱따구리.

![Downy Woodpecker](./CUB_200_2011/images/192.Downy_Woodpecker/Downy_Woodpecker_0001_184484.jpg)

---

### 200. Common Yellowthroat (황금목 휘파람새)
> 수컷은 노란 목·배, 검은 안면 마스크가 특징인 소형 명금류.

![Common Yellowthroat](./CUB_200_2011/images/200.Common_Yellowthroat/Common_Yellowthroat_0003_190521.jpg)

---

## 5. 어노테이션 상세

### 5-1. 바운딩 박스 (Bounding Box)

모든 이미지에 조류를 감싸는 단일 바운딩 박스가 제공된다.

```
# 형식: <image_id> <x> <y> <width> <height>  (픽셀 단위)
1  60.0  27.0  325.0  304.0
2 139.0  30.0  153.0  264.0
3  14.0 112.0  388.0  186.0
```

### 5-2. 부위 위치 (Part Locations)

15개 신체 부위에 대한 픽셀 좌표와 가시성(visible) 정보가 제공된다.

| Part ID | 부위 이름 |
|---------|---------|
| 1 | back (등) |
| 2 | beak (부리) |
| 3 | belly (배) |
| 4 | breast (가슴) |
| 5 | crown (정수리) |
| 6 | forehead (이마) |
| 7 | left eye (왼쪽 눈) |
| 8 | left leg (왼쪽 다리) |
| 9 | left wing (왼쪽 날개) |
| 10 | nape (목덜미) |
| 11 | right eye (오른쪽 눈) |
| 12 | right leg (오른쪽 다리) |
| 13 | right wing (오른쪽 날개) |
| 14 | tail (꼬리) |
| 15 | throat (목) |

```
# 형식: <image_id> <part_id> <x> <y> <visible>
# 총 176,820행 (11,788 이미지 × 15 부위)
```

### 5-3. 속성 레이블 (Attribute Labels)

312개의 이진 속성이 MTurk 작업자들에 의해 레이블링되었다. 속성 카테고리 예시:

| 카테고리 | 속성 예시 |
|---------|---------|
| 부리 형태 | curved, dagger, hooked, needle, spatulate, cone... |
| 날개 색상 | blue, brown, iridescent, grey, yellow, green, orange... |
| 윗꼬리 색상 | buff, white, black, red, brown... |
| 가슴 패턴 | solid, striped, multi-colored, spotted... |
| 몸 크기 | small, medium, large, very_large |
| 깃털 유형 | iridescent, crested |

```
# 클래스별 연속 속성값 (0~100, 해당 속성이 나타나는 비율)
# 형식: 200행 × 312열 행렬 (class_attribute_labels_continuous.txt)
```

---

## 6. 학습/테스트 분할

| 분할 | 이미지 수 | 비율 |
|------|---------|------|
| Train | 5,994 | 50.8% |
| Test | 5,794 | 49.2% |
| **Total** | **11,788** | **100%** |

- 각 클래스에서 약 절반씩 train/test로 분할
- `train_test_split.txt`에서 `1 = 학습`, `0 = 테스트`

---

## 7. FGVC 분야에서의 난이도

CUB-200-2011이 어려운 이유:

1. **클래스 간 유사성**: 같은 속(genus)의 새들은 색상, 형태가 매우 유사함
2. **클래스 내 다양성**: 조명, 자세, 배경, 촬영 각도가 모두 다름
3. **클래스당 적은 샘플**: 클래스당 평균 약 30장(학습 기준)의 이미지만 존재
4. **소형 판별 특징**: 부리 색, 눈 테두리, 날개 끝 패턴 등 미세한 부분이 핵심

### 주요 벤치마크 성능 (Top-1 Accuracy)

| 모델 | 정확도 | 특징 |
|------|--------|------|
| ResNet-50 (baseline) | ~84% | 일반 분류 모델 |
| ViT-B/16 (fine-tuned) | ~88% | 비전 트랜스포머 기본 |
| TransFG (ViT-B/16) | ~91.7% | 부위 선택적 어텐션 |
| SOTA (2023~) | ~93%+ | 다양한 어텐션 기법 |

---

## 8. TransFG와의 연관성

TransFG(Transformer for Fine-Grained Recognition)는 CUB-200-2011을 핵심 벤치마크로 사용한다:

- **입력**: 448×448 리사이즈 후 ViT patch 임베딩
- **핵심 아이디어**: ViT의 어텐션 맵에서 판별력 높은 패치를 선택적으로 활용
- **바운딩 박스**: 학습 시 crop에 활용 가능 (선택적)
- **부위 어노테이션**: 어텐션과 부위 정렬 연구에 활용

```python
# project_config.py에서의 CUB 경로 설정
DATA_DIR = "/Users/macminim4/Aiffel03/Project01/TransFG/CUB_200_2011"
NUM_CLASSES = 200
IMG_SIZE = 448
```

---

## 9. 인용 정보

```bibtex
@techreport{WelinderEtal2010,
  Author = {P. Welinder and S. Branson and T. Mita and C. Wah and
            F. Schroff and S. Belongie and P. Perona},
  Institution = {California Institute of Technology},
  Number = {CNS-TR-2010-001},
  Title = {{Caltech-UCSD Birds 200}},
  Year = {2010}
}
```

---

*데이터셋 공식 페이지: http://www.vision.caltech.edu/visipedia*
