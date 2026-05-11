# TransFG 스터디 — CUB-200-2011 새 200종 분류

> **논문**: TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI 2022)
> **arXiv**: https://arxiv.org/abs/2103.07976
> **목표**: ViT 기반 Part Selection Module을 이용해 새 200종을 분류하는 Fine-Grained 분류 실습

---

## 핵심 아이디어

```
일반 ViT                     TransFG
─────────────────            ─────────────────────────────
이미지 → 패치 분할           이미지 → 패치 분할
→ Transformer 인코더         → Transformer 인코더
→ [CLS] 토큰으로 분류        → Part Selection Module (중요 패치 선별)
                             → 선별된 패치 + [CLS] 로 분류
```

새의 부리, 날개, 발 등 **핵심 부위(Part)** 에 attention이 집중되어 미세한 차이를 구분합니다.

![TransFG Architecture](../TransFG.png)

---

## 프로젝트 구조

```
TransFG/
├── TransFG_CUB200.ipynb     # 메인 실습 노트북 (Cell 01~20)
├── train.py                 # 원본 학습 스크립트 (multi-GPU)
├── train_utils.py           # 수정된 학습 유틸 (단일 디바이스, apex 제거)
├── project_config.py        # 경로/하이퍼파라미터 설정
├── models/
│   ├── configs.py           # ViT 모델 설정 (ViT-B_16 등)
│   └── modeling.py          # TransFG 모델 구현 (Part Selection 포함)
├── utils/
│   ├── dataset.py           # CUB-200-2011 데이터셋 로더
│   ├── data_utils.py        # DataLoader 생성 유틸
│   ├── autoaugment.py       # 데이터 증강
│   ├── scheduler.py         # WarmupCosineSchedule
│   └── dist_util.py        # 분산 학습 유틸 (단일 GPU용으로 비활성화)
├── code_study_md/           # 코드별 상세 설명 (초보자용)
├── teaching_md/             # 다른 데이터셋 적용 가이드
├── for_git/
│   ├── README_STUDY.md      # 이 파일
│   ├── SETUP_GUIDE.md       # 환경 설정 가이드
│   ├── GIT_HOWTO.md         # git 사용 가이드
│   └── images/              # 문서용 이미지
├── requirements_updated.txt # 수정된 패키지 목록
└── setup_env.sh             # 가상환경 설정 스크립트
```

---

## 빠른 시작

### 1. 저장소 클론

```bash
git clone <repo-url>
cd TransFG
```

### 2. 환경 설정

```bash
# 가상환경 생성
python3.9 -m venv TransFG_venv
source TransFG_venv/bin/activate

# 패키지 설치
pip install -r requirements_updated.txt
```

### 3. 데이터 & 사전학습 가중치 다운로드

> 용량 문제로 git에 포함되지 않습니다. 아래 명령어로 직접 다운로드하세요.

```bash
# CUB-200-2011 데이터셋 (약 1.2GB)
# https://www.vision.caltech.edu/datasets/cub-200-2011/ 에서 수동 다운로드 후
# TransFG/CUB_200_2011/ 에 압축 해제

# ViT-B_16 사전학습 가중치 (약 394MB)
mkdir -p pretrained
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P pretrained/
```

### 4. 노트북 실행

```bash
jupyter notebook TransFG_CUB200.ipynb
```

---

## 노트북 셀 구성 (Cell 01~20)

| Cell | 내용 | 비고 |
|------|------|------|
| 01 | 환경 확인 (Python, PyTorch, Device) | MPS/CUDA/CPU 자동 감지 |
| 02 | 패키지 설치 | 최초 1회만 |
| 03 | 데이터셋 다운로드 안내 | 수동 다운로드 필요 |
| 04 | Pretrained 가중치 다운로드 | wget 사용 |
| 05 | 모듈 임포트 및 경로 설정 | |
| 06 | 데이터셋 탐색 (클래스, 이미지 수) | |
| 07 | 샘플 이미지 시각화 | |
| 08 | 하이퍼파라미터 설정 | |
| 09 | DataLoader 생성 | |
| 10 | TransFG 모델 생성 | ViT-B_16 기반 |
| 11 | 모델 구조 탐색 | Part Selection Module 확인 |
| 12 | 빠른 학습 테스트 (200 steps) | 동작 확인용 |
| 13 | 학습 history 시각화 | loss/acc 그래프 |
| 14 | 전체 학습 (10,000 steps) | 주석 처리됨 |
| 15 | 체크포인트 로드 | |
| 16 | 테스트셋 평가 | |
| 17 | 단일 이미지 추론 | |
| 18 | 배치 예측 시각화 | |
| 19 | Self-Attention Map 시각화 | 모델이 어디를 보는지 확인 |
| 20 | 12개 Head Attention 비교 | Multi-head attention 시각화 |

---

## 수정 사항 (원본 대비)

| 항목 | 원본 | 수정 |
|------|------|------|
| imread | `scipy.misc.imread` | PIL + numpy |
| 혼합정밀도 | apex FP16 | torch.cuda.amp (CUDA) / FP32 (MPS) |
| 분산학습 | `dist.barrier()` | 단일 디바이스 (제거) |
| pin_memory | True | CUDA=True, MPS/CPU=False |
| num_workers | 4 | macOS=0, Linux=4 |

---

## 디바이스별 학습 시간 예상

| 디바이스 | 학습 시간 (10,000 steps) |
|---------|------------------------|
| NVIDIA RTX 3090 (CUDA) | ~1.5~2.5시간 |
| Apple M4 (MPS) | ~15~25시간 |
| CPU | 수십 시간 이상 |

---

## 참고 자료

- [논문 원본 (arXiv)](https://arxiv.org/abs/2103.07976)
- [공식 GitHub](https://github.com/TACJu/TransFG)
- [ViT 원논문](https://arxiv.org/abs/2010.11929)
- [CUB-200-2011 데이터셋](https://www.vision.caltech.edu/datasets/cub-200-2011/)
- 상세 설명: `code_study_md/TransFG_CUB_teaching.md`
- 학습 가이드: `STUDY_GUIDE.md`
