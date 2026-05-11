# TransFG: A Transformer Architecture for Fine-grained Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=transfg-a-transformer-architecture-for-fine)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=transfg-a-transformer-architecture-for-fine)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=transfg-a-transformer-architecture-for-fine)

**Paper URL**: [https://arxiv.org/abs/2103.07976](https://arxiv.org/abs/2103.07976)
**Original Repository**: [TACJu/TransFG](https://github.com/TACJu/TransFG)

TransFG proposes a Part Selection Module on top of ViT (Vision Transformer) to select discriminative image patches, achieving **91.7%** on CUB-200-2011 and **90.8%** on NABirds.

![framework](./TransFG.png)

---

# 👁️ Eye-Opening Project

## 🚀 Phase 1 Objective: Sequential Transfer Learning
The primary goal of Phase 1 is to verify the generalizability and knowledge transfer capability of the TransFG model across diverse fine-grained domains.

**Transfer Learning Pipeline:**
```text
ImageNet Pretraining (ViT-B_16, 21k classes)
        ↓
CUB-200 Fine-tuning (200 classes)       ← 이 레포
        ↓ [Checkpoint: CUB Pretrained Model]
Aircraft Fine-tuning (100 classes)
        ↓ [Checkpoint: Aircraft Pretrained Model]
Stanford Cars Fine-tuning (196 classes)
```

---

## 1. Setup & Installation

### 1.1. Install Dependencies
```zsh
# 가상환경 생성 및 활성화
python3.9 -m venv TransFG_venv
source TransFG_venv/bin/activate

# 패키지 설치
pip install -r requirements_updated.txt
```

> [!NOTE]
> **macOS Support**: This project supports training on Apple Silicon (MPS). However, training is ~10-20x slower than NVIDIA GPU. For full training (10,000 steps), an NVIDIA GPU environment is recommended.

### 1.2. Datasets & Pretrained Models
- **CUB-200-2011**: [Download Link](https://www.vision.caltech.edu/datasets/cub-200-2011/)
  Place extracted folder at `CUB_200_2011/` in the project root.

#### 📥 Download Pretrained Weights (ViT-B_16)
```zsh
mkdir -p pretrained
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P pretrained/
```

---

## 2. 🏋️ Training

### Option A: Jupyter Notebook (권장)
```zsh
jupyter notebook TransFG_CUB200.ipynb
```
- Cell 01~12: 환경 확인 → 데이터 로드 → 모델 생성 → 빠른 테스트 (200 steps)
- Cell 14: 전체 학습 (10,000 steps)

### Option B: Command Line
```zsh
python train.py \
  --dataset CUB_200_2011 \
  --split overlap \
  --num_steps 10000 \
  --name transfg_cub200
```
- **Configuration**: Modify `project_config.py` for hyperparameters.
- **Checkpointing**: Models are saved in `./output/{name}_checkpoint.bin`.

---

## 3. 📊 Evaluation & Inference

### Option A: Quick Accuracy Check (via Notebook)
```zsh
jupyter notebook TransFG_CUB200.ipynb
# Cell 15-16 실행: 체크포인트 로드 → 테스트셋 평가
```
- **Output**: Top-1 Accuracy on CUB-200-2011 test set.

### Option B: Single Image & Batch Inference
```zsh
jupyter notebook TransFG_CUB200.ipynb
# Cell 17: 단일 이미지 추론
# Cell 18: 배치 예측 시각화
```

#### 💡 Comparison Table
| Feature | Notebook (Cell 16) | Notebook (Cell 17-18) |
| :--- | :--- | :--- |
| **Test Accuracy** | ✅ Yes | ❌ No |
| **Per-image Prediction** | ❌ No | ✅ Yes |
| **Visualization** | ❌ No | ✅ Yes |

---

## 4. 🔥 Attention Map Visualization
Visualize where the model focuses using Self-Attention Maps.

```zsh
jupyter notebook TransFG_CUB200.ipynb
# Cell 19: Self-Attention Map 시각화
# Cell 20: 12개 Head Attention 비교
```

---

## 🛠️ Modifications from Original

This fork applies the following changes for macOS/MPS compatibility and single-device training:

| File | Change |
| :--- | :--- |
| `utils/dataset.py` | `scipy.misc.imread` → PIL + numpy |
| `utils/data_utils.py` | `InterpolationMode.BILINEAR`, `pin_memory=False` on MPS |
| `train_utils.py` | apex/DDP 제거, `torch.cuda.amp` 기반 단일 디바이스 학습 |
| `project_config.py` | CUDA → MPS → CPU 자동 감지, 통합 하이퍼파라미터 설정 |
| `TransFG_CUB200.ipynb` | Cell 01~20 전체 실습 노트북 |

---

## Citation

```bibtex
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```

---

### Acknowledgement
- Many thanks to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) for the PyTorch reimplementation of [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
- Study materials and macOS compatibility patches by [Forge-AI-Core](https://github.com/Forge-AI-Core).
