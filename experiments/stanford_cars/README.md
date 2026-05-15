# TransFG Stanford Cars — Phase 2 Fine-tuning Experiment

> **Fine-Grained Visual Categorization (FGVC) on Stanford Cars**
> Aircraft 사전학습 가중치를 시작점으로 한 자동차 196 모델 분류 실험

**Paper**: [TransFG (arxiv:2103.07976)](https://arxiv.org/abs/2103.07976)
**Original Repository**: [TACJu/TransFG](https://github.com/TACJu/TransFG)
**Dataset**: [Stanford Cars (Krause et al., 2013)](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) — HuggingFace [tanganke/stanford_cars](https://huggingface.co/datasets/tanganke/stanford_cars)

---

## 🚀 Phase 2 Context — Sequential Transfer Learning

본 실험은 TransFG 모델의 **도메인 유사도 기반 누적 전이학습** 가설을 검증하는 Phase 2 실험입니다:

```text
ImageNet Pretraining (ViT-B_16, 21k classes)
        ↓
CUB-200 Fine-tuning (200 classes)             ← master 브랜치 (조류, 유기체)
        ↓ [Checkpoint: cub_pretrained.bin]
Aircraft Fine-tuning (100 classes)            ← Phase 1 (항공기, 인공물)
        ↓ [Checkpoint: aircraft_pretrained.bin]
Stanford Cars Fine-tuning (196 classes)       ← 🚗 이번 실험 (자동차, 인공물)
```

### 🎯 핵심 가설

> **Fine-grained 전이학습에서 도메인 유사도가 절대 정확도보다 중요할 것이다.**

| | CUB checkpoint | **Aircraft checkpoint** |
|---|---|---|
| 도메인 | 조류 (유기체) | **항공기 (인공물)** ← Cars와 가장 유사 |
| 텍스처 | 깃털·곡선 | **금속·직선** ← Cars와 동일 |
| 식별 단위 | 부리·날개·깃털 | **동체·날개·꼬리** ← 차체와 유사 |
| 모델 정확도 | 90.84% | 78.97% |

### 🎯 실험 구성

본 디렉토리에는 **두 가지 실험**이 포함되어 있습니다:

| 실험 | 시작 가중치 | Steps | 출력 | 목적 |
|---|---|---|---|---|
| **A. ImageNet baseline** | `ViT-B_16.npz` | 200 (sanity check) | `output/stanford_cars/` | 빠른 baseline 비교용 (1.37% 기준점) |
| **B. Aircraft→Cars** ⭐ | `aircraft_pretrained.bin` | 본격 학습 | `output/stanford_cars_from_aircraft/` | **Phase 2 메인 실험** |

> 💡 노트북 셀 19의 결론 임계값: `+5%p` 이상 향상이면 도메인 유사 전이학습 효과 강력 확인.

---

## 📂 Directory Structure

```
experiments/stanford_cars/
├── README.md                          # (이 문서)
├── Stanford_Cars_Dataset_Review.md    # Stanford Cars 데이터셋 상세 리뷰
├── TransFG_Stanford_Cars_New.ipynb    # ⭐ 메인 실습 노트북
├── project_config.py                  # 통합 설정 (경로/디바이스/하이퍼파라미터)
├── setup_env.sh                       # 환경 설치 스크립트
├── requirements.txt                   # 의존성 (CUDA 13.0 / aarch64 지원)
│
├── data_loader_cars.py                # Stanford Cars DataLoader (parquet 기반)
├── dataset_stanford_cars.py           # Stanford Cars Dataset 클래스
├── trainer.py                         # 학습 루프 (AMP, single-device)
├── inference_utils.py                 # 추론 헬퍼 (단일/배치 예측, 평가)
├── visualization.py                   # 시각화 (Attention Map, 예측 등)
├── run_cars_train.py                  # CLI 학습 스크립트
│
├── models/                            # ViT-B/16 + TransFG 모델 코드
│   ├── __init__.py
│   ├── configs.py                     # 모델 설정
│   └── modeling.py                    # VisionTransformer 구현
│
├── Stanford_Cars/                     # 🚫 데이터 (.gitignore — 별도 다운로드)
├── pretrained/                        # 🚫 사전학습 가중치 (.gitignore)
├── output/                            # 🚫 학습 결과 (.gitignore)
└── logs/                              # 🚫 TensorBoard 로그 (.gitignore)
```

---

## 1. Setup & Installation

### 1.1. 환경 설치 (자동 분기)

```bash
cd experiments/stanford_cars
bash setup_env.sh
source TransFG_cars_venv/bin/activate
```

`setup_env.sh`는 환경을 자동 감지해서 적절한 PyTorch를 설치합니다:

| 환경 | PyTorch |
|---|---|
| macOS | CPU/MPS |
| Linux aarch64 (NVIDIA GB10 등) | **CUDA 13.0** (cu130) |
| Linux x86_64 | CUDA 12.1 (cu121) |

### 1.2. 데이터 준비 (Stanford Cars — parquet 형식)

Stanford Cars는 원본 공식 사이트의 다운로드가 불안정해, **HuggingFace의 `tanganke/stanford_cars`** 미러를 사용합니다 (parquet 형식, 이미지 bytes 내장).

**옵션 A: `huggingface-cli` 사용 (권장)**
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tanganke/stanford_cars \
  --repo-type dataset \
  --local-dir ./Stanford_Cars
```

**옵션 B: `git clone` (LFS 필요)**
```bash
git lfs install
git clone https://huggingface.co/datasets/tanganke/stanford_cars Stanford_Cars
```

설치 후 `Stanford_Cars/data/` 안에 `train-*.parquet`, `test-*.parquet` 파일이 있어야 합니다.

> **데이터셋 상세**: [Stanford_Cars_Dataset_Review.md](./Stanford_Cars_Dataset_Review.md) 참고

### 1.3. 사전학습 가중치 준비

`project_config.py`는 가중치를 다음 우선순위로 자동 선택합니다:

```python
PRETRAINED_WEIGHTS = AIRCRAFT_PRETRAINED if exists else IMAGENET_WEIGHTS
```

**옵션 A: ImageNet baseline (sanity check)**
```bash
mkdir -p pretrained
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P pretrained/
```

**옵션 B: Aircraft→Cars (Phase 2 메인 실험, 권장)** ⭐
```bash
# (a) feature/SCRUM-24 브랜치의 TransFG_Aircraft_New.ipynb로 직접 학습 후
#     output/transfg_aircraft_from_cub_checkpoint.bin 을 pretrained/aircraft_pretrained.bin 으로 복사
cp ../aircraft/output/fgvc_aircraft_from_cub/transfg_aircraft_from_cub_checkpoint.bin \
   ./pretrained/aircraft_pretrained.bin

# (b) 또는 팀 공유 드라이브에서 받기 (팀원에게 문의)
```

---

## 2. 🏋️ Training

### Option A: Jupyter Notebook (권장)

```bash
jupyter notebook TransFG_Stanford_Cars_New.ipynb
```

노트북 구성:
- **Cell 01~03**: 환경 확인 (디바이스, 패키지 import)
- **Cell 04~07**: 데이터 로드 (Stanford Cars parquet → DataLoader)
- **Cell 08~11**: 하이퍼파라미터 + DataLoader 생성
- **Cell 12~13**: 모델 생성 (ViT-B/16 + Aircraft 가중치 로드) ⭐
- **Cell 14~17**: 빠른 학습 테스트 (200 steps) + 학습 곡선
- **Cell 18~19**: ImageNet-21k 베이스라인과 비교 ⭐
- **Cell 20~21**: 전체 학습 코드 (Optional, 10,000 steps, 주석 처리)

### Option B: Command Line

```bash
python run_cars_train.py --steps 10000
# 중단 후 이어서:
python run_cars_train.py --steps 10000 --resume
```

- **Configuration**: 하이퍼파라미터는 `project_config.py` 수정
- **Checkpointing**: 학습 결과는 `./output/{name}_checkpoint.bin`

---

## 3. 📊 Evaluation & Inference

### Test Accuracy (Top-1)
```bash
jupyter notebook TransFG_Stanford_Cars_New.ipynb
# Cell 18-19 실행: 체크포인트 로드 → Stanford Cars 테스트셋 평가, ImageNet baseline(1.37%)과 비교
```

### Single Image / Batch Inference
```python
from inference_utils import predict_batch, evaluate_dataset

# 배치 단위 예측
preds, probs = predict_batch(model, x_batch, DEVICE)

# 전체 테스트셋 평가 (accuracy + 모든 prediction 반환)
result = evaluate_dataset(model, test_loader, DEVICE)
print(result["accuracy"])
```

### Attention Map Visualization
```python
from visualization import visualize_attention

visualize_attention(model, img_tensor, DEVICE, head=0)
# → 자동차의 헤드라이트·그릴·휠 영역에 attention 집중 여부 확인 가능
```

---

## 4. 🛠️ Modifications from Aircraft Experiment

Aircraft 실험(SCRUM-24)과 비교한 변경점:

| 항목 | Aircraft | **Stanford Cars** |
|---|---|---|
| 데이터셋 | FGVC-Aircraft (항공기 100종) | Stanford Cars (자동차 196종) |
| `NUM_CLASSES` | 100 | **196** |
| 시작 가중치 | CUB 학습 결과 (`cub_pretrained.bin`) | **Aircraft 학습 결과** (`aircraft_pretrained.bin`) |
| 누적 전이 단계 | ImageNet → CUB → Aircraft (2단계) | **ImageNet → CUB → Aircraft → Cars (3단계)** |
| 데이터 형식 | 텍스트 매핑 + 이미지 폴더 | **parquet (이미지 bytes 내장)** |
| 분할 | train/val/test (3분할) | **train/test (2분할)** |
| Dataset 클래스 | `dataset_aircraft.py` | `dataset_stanford_cars.py` (parquet 처리) |
| DataLoader | `data_loader_aircraft.py` | `data_loader_cars.py` (+ prefetch_factor, persistent_workers) |
| 학습 스크립트 | `run_aircraft_train.py` | `run_cars_train.py` |
| Baseline (200 step) | val_acc 6.57% | **test_acc 1.37%** |
| 결론 임계값 | ±2%p | **+5%p** |

> 💡 본 실험은 Aircraft와 **완전 독립된 모듈**로 구성되어 있어, Cars 코드 수정이 Aircraft/CUB 실험에 영향을 주지 않습니다.

---

## 5. 📚 Citation

```bibtex
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}

@inproceedings{krause2013collecting,
  title={3D Object Representations for Fine-Grained Categorization},
  author={Krause, Jonathan and Stark, Michael and Deng, Jia and Fei-Fei, Li},
  booktitle={4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13)},
  year={2013}
}
```

---

## 🙏 Acknowledgement

- Original TransFG: [TACJu/TransFG](https://github.com/TACJu/TransFG)
- ViT-pytorch: [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- Stanford Cars: [Krause et al. (Stanford)](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- Stanford Cars (HuggingFace mirror): [tanganke/stanford_cars](https://huggingface.co/datasets/tanganke/stanford_cars)
- Phase 2 study & macOS/aarch64 compatibility: [Forge-AI-Core](https://github.com/Forge-AI-Core)

---

## 📌 Related

- **Jira Ticket**: SCRUM-35
- **Branch**: `feature/SCRUM-35`
- **Previous Phase**: Aircraft fine-tuning (`feature/SCRUM-24`, `experiments/aircraft/`)
- **Pipeline Lineage**: ImageNet → CUB (master) → Aircraft (SCRUM-24) → **Cars (SCRUM-35)** ⭐
