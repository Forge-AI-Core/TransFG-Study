# TransFG Aircraft — Phase 1 Fine-tuning Experiment

> **Fine-Grained Visual Categorization (FGVC) on FGVC-Aircraft**
> CUB-200 사전학습 가중치를 시작점으로 한 항공기 100종 분류 실험

**Paper**: [TransFG (arxiv:2103.07976)](https://arxiv.org/abs/2103.07976)
**Original Repository**: [TACJu/TransFG](https://github.com/TACJu/TransFG)
**Dataset**: [FGVC-Aircraft (Oxford VGG, 2013)](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

---

## 🚀 Phase 1 Context — Sequential Transfer Learning

본 실험은 TransFG 모델의 도메인 일반화 능력을 검증하기 위한 **Phase 1 파이프라인의 두 번째 단계**입니다:

```text
ImageNet Pretraining (ViT-B_16, 21k classes)
        ↓
CUB-200 Fine-tuning (200 classes)           ← master 브랜치
        ↓ [Checkpoint: cub_pretrained.bin]
Aircraft Fine-tuning (100 classes)          ← 🛬 이번 실험 (이 폴더)
        ↓ [Checkpoint: aircraft_pretrained.bin]
Stanford Cars Fine-tuning (196 classes)     ← 다음 단계 (예정)
```

### 🎯 실험 구성

본 디렉토리에는 **두 가지 실험**이 포함되어 있습니다:

| 실험 | 시작 가중치 | Steps | 출력 | 목적 |
|---|---|---|---|---|
| **A. ImageNet baseline** | `ViT-B_16.npz` | 200 (sanity check) | `output/fgvc_aircraft/` | 빠른 baseline 비교용 |
| **B. CUB→Aircraft** ⭐ | `cub_pretrained.bin` | 본격 학습 | `output/fgvc_aircraft_from_cub/` | **Phase 1 메인 실험** |

> 💡 핵심 가설: CUB로 학습된 fine-grained 분류 능력이 Aircraft 도메인에도 전이되는가?

---

## 📂 Directory Structure

```
experiments/aircraft/
├── README.md                       # (이 문서)
├── Aircraft_Dataset_Review.md      # FGVC-Aircraft 데이터셋 상세 리뷰
├── TransFG_Aircraft_New.ipynb      # ⭐ 메인 실습 노트북
├── project_config.py               # 통합 설정 (경로/디바이스/하이퍼파라미터)
├── setup_env.sh                    # 환경 설치 스크립트
├── requirements.txt                # 의존성 (CUDA 13.0 / aarch64 지원)
│
├── data_loader_aircraft.py         # Aircraft DataLoader
├── dataset_aircraft.py             # Aircraft Dataset 클래스
├── trainer.py                      # 학습 루프 (AMP, single-device)
├── inference_utils.py              # 추론 헬퍼 (단일/배치 예측, 평가)
├── visualization.py                # 시각화 (Attention Map, 예측 등)
├── run_aircraft_train.py           # CLI 학습 스크립트
│
├── models/                         # ViT-B/16 + TransFG 모델 코드
│   ├── __init__.py
│   ├── configs.py                  # 모델 설정
│   └── modeling.py                 # VisionTransformer 구현
│
├── FGVC-Aircraft/                  # 🚫 데이터 (.gitignore — 별도 다운로드)
├── pretrained/                     # 🚫 사전학습 가중치 (.gitignore)
├── output/                         # 🚫 학습 결과 (.gitignore)
└── logs/                           # 🚫 TensorBoard 로그 (.gitignore)
```

---

## 1. Setup & Installation

### 1.1. 환경 설치 (자동 분기)

```bash
cd experiments/aircraft
bash setup_env.sh
source TransFG_aircraft_venv/bin/activate
```

`setup_env.sh`는 환경을 자동 감지해서 적절한 PyTorch를 설치합니다:

| 환경 | PyTorch |
|---|---|
| macOS | CPU/MPS |
| Linux aarch64 (NVIDIA GB10 등) | **CUDA 13.0** (cu130) |
| Linux x86_64 | CUDA 12.1 (cu121) |

### 1.2. 데이터 준비 (FGVC-Aircraft)

```bash
mkdir -p FGVC-Aircraft
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xzf fgvc-aircraft-2013b.tar.gz -C ./FGVC-Aircraft
rm fgvc-aircraft-2013b.tar.gz
```

> **데이터셋 상세**: [Aircraft_Dataset_Review.md](./Aircraft_Dataset_Review.md) 참고

### 1.3. 사전학습 가중치 준비

`project_config.py`는 가중치를 다음 우선순위로 자동 선택합니다:

```python
PRETRAINED_WEIGHTS = CUB_PRETRAINED if exists else IMAGENET_WEIGHTS
```

**옵션 A: ImageNet baseline (sanity check)**
```bash
mkdir -p pretrained
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P pretrained/
```

**옵션 B: CUB→Aircraft (Phase 1 메인 실험, 권장)** ⭐
```bash
# (a) master 브랜치의 TransFG_CUB200.ipynb로 직접 학습 후
#     output/transfg_cub200_checkpoint.bin 을 pretrained/cub_pretrained.bin 으로 복사
cp ../../output/transfg_cub200_checkpoint.bin ./pretrained/cub_pretrained.bin

# (b) 또는 팀 공유 드라이브에서 받기 (팀원에게 문의)
```

---

## 2. 🏋️ Training

### Option A: Jupyter Notebook (권장)

```bash
jupyter notebook TransFG_Aircraft_New.ipynb
```

노트북 구성:
- **Cell 01~03**: 환경 확인 (디바이스, 패키지 import)
- **Cell 04~06**: 데이터 로드 (FGVC-Aircraft → DataLoader)
- **Cell 07~09**: 모델 생성 (ViT-B/16 + 사전학습 가중치 로드)
- **Cell 10~12**: 빠른 검증 (200 steps - ImageNet baseline)
- **Cell 13~15**: 본격 학습 (CUB pretrained 시작)
- **Cell 16~17**: 평가 (테스트셋 Top-1 Accuracy)
- **Cell 18~19**: 추론 시각화 (단일/배치)
- **Cell 20~21**: Attention Map 시각화

### Option B: Command Line

```bash
python run_aircraft_train.py \
  --dataset FGVC-Aircraft \
  --num_steps 10000 \
  --name transfg_aircraft_from_cub \
  --pretrained_dir ./pretrained/cub_pretrained.bin
```

- **Configuration**: 하이퍼파라미터는 `project_config.py` 수정
- **Checkpointing**: 학습 결과는 `./output/{name}_checkpoint.bin`

---

## 3. 📊 Evaluation & Inference

### Test Accuracy (Top-1)
```bash
jupyter notebook TransFG_Aircraft_New.ipynb
# Cell 16-17 실행: 체크포인트 로드 → FGVC-Aircraft 테스트셋 평가
```

### Single Image / Batch Inference
```bash
jupyter notebook TransFG_Aircraft_New.ipynb
# Cell 18-19 실행: 단일 이미지 예측, 배치 시각화
```

### Attention Map Visualization
```bash
jupyter notebook TransFG_Aircraft_New.ipynb
# Cell 20-21 실행: Self-Attention Map, 12-Head 비교
```

---

## 4. 🛠️ Modifications from CUB Experiment

CUB 실험(master 브랜치)과 비교한 변경점:

| 항목 | CUB | Aircraft |
|---|---|---|
| 데이터셋 | CUB-200-2011 (조류 200종) | FGVC-Aircraft (항공기 100종) |
| `NUM_CLASSES` | 200 | **100** |
| 시작 가중치 | ImageNet (`ViT-B_16.npz`) | **CUB 학습 결과** (`cub_pretrained.bin`) |
| DataLoader | `utils/data_utils.py` | `data_loader_aircraft.py` (별도 모듈) |
| Dataset 클래스 | `utils/dataset.py` | `dataset_aircraft.py` (별도 모듈) |
| 환경 | CUDA 12.1 (x86_64) | **CUDA 13.0** (aarch64 지원 추가) |
| 학습 스크립트 | `train.py` | `run_aircraft_train.py` |

> 💡 본 실험은 CUB와 **완전 독립된 모듈**로 구성되어 있어, Aircraft 코드 수정이 CUB 실험에 영향을 주지 않습니다.

---

## 5. 📚 Citation

```bibtex
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}

@techreport{maji2013fine,
  title={Fine-grained visual classification of aircraft},
  author={Maji, Subhransu and Rahtu, Esa and Kannala, Juho and Blaschko, Matthew and Vedaldi, Andrea},
  institution={arXiv:1306.5151},
  year={2013}
}
```

---

## 🙏 Acknowledgement

- Original TransFG: [TACJu/TransFG](https://github.com/TACJu/TransFG)
- ViT-pytorch: [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- FGVC-Aircraft: [Maji et al. (Oxford VGG)](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- Phase 1 study & macOS/aarch64 compatibility: [Forge-AI-Core](https://github.com/Forge-AI-Core)

---

## 📌 Related

- **Jira Ticket**: SCRUM-24
- **Branch**: `feature/SCRUM-24`
- **Previous Phase**: CUB-200 fine-tuning (master 브랜치)
- **Next Phase**: Stanford Cars fine-tuning (예정)
