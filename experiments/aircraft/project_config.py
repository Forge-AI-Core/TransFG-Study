# coding=utf-8
"""
project_config.py
TransFG Aircraft 프로젝트 공통 설정 (Phase 1 - CUB→Aircraft 전이학습)
- 경로 자동 감지 (FGVC-Aircraft 폴더 구조 처리)
- 디바이스 자동 감지: CUDA -> MPS -> CPU
- 학습 하이퍼파라미터
- make_args(): 기존 utils 코드와 호환되는 Namespace 생성
"""
import os
import platform
import argparse
import torch

# ── 경로 설정 ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# FGVC-Aircraft: tarball 해제시 중첩 폴더 구조 자동 감지
# experiments/aircraft/FGVC-Aircraft/fgvc-aircraft-2013b/ 또는 experiments/aircraft/FGVC-Aircraft/
_aircraft_base = os.path.join(PROJECT_ROOT, 'FGVC-Aircraft')
DATA_ROOT = (
    os.path.join(_aircraft_base, 'fgvc-aircraft-2013b')
    if os.path.isdir(os.path.join(_aircraft_base, 'fgvc-aircraft-2013b'))
    else _aircraft_base
)

# ── 가중치 설정 (Phase 1: CUB→Aircraft 전이학습) ──────────────
PRETRAINED_DIR   = os.path.join(PROJECT_ROOT, 'pretrained')
IMAGENET_WEIGHTS = os.path.join(PRETRAINED_DIR, 'ViT-B_16.npz')
CUB_PRETRAINED   = os.path.join(PRETRAINED_DIR, 'cub_pretrained.bin')

# 기본 가중치: CUB 학습 결과 우선 사용 (Phase 1 핵심)
# CUB 가중치가 없으면 ImageNet으로 폴백 (baseline 비교용)
PRETRAINED_WEIGHTS = CUB_PRETRAINED if os.path.isfile(CUB_PRETRAINED) else IMAGENET_WEIGHTS

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
LOG_DIR    = os.path.join(PROJECT_ROOT, 'logs')


# ── 디바이스 감지: CUDA -> MPS -> CPU ─────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()
_dt = DEVICE.type

# ── 배치 크기 (디바이스별 권장값) ─────────────────────────────
TRAIN_BATCH_SIZE = {'cuda': 16, 'mps': 8, 'cpu': 4}.get(_dt, 4)
EVAL_BATCH_SIZE  = {'cuda': 8,  'mps': 4, 'cpu': 2}.get(_dt, 2)

# ── DataLoader 설정 ───────────────────────────────────────────
NUM_WORKERS = 0 if platform.system() == 'Darwin' else 4
PIN_MEMORY  = (_dt == 'cuda')

# ── 모델 / 데이터 설정 ────────────────────────────────────────
DATASET     = 'FGVC-Aircraft'
NUM_CLASSES = 100                # FGVC-Aircraft: 100 variants (model variants)
MODEL_TYPE  = 'ViT-B_16'
IMG_SIZE    = 448
SPLIT       = 'non-overlap'      # 'non-overlap' | 'overlap'
SLIDE_STEP  = 12

# ── 학습 하이퍼파라미터 ───────────────────────────────────────
NUM_STEPS     = 10000
WARMUP_STEPS  = 500
EVAL_EVERY    = 100
LR            = 3e-2
WEIGHT_DECAY  = 0.0
MAX_GRAD_NORM = 1.0
DECAY_TYPE    = 'cosine'         # 'cosine' | 'linear'
SMOOTHING     = 0.0
SEED          = 42
GRAD_ACCUM    = 1


def make_args(**overrides) -> argparse.Namespace:
    """
    기존 trainer.py, data_loader_aircraft.py 등과 호환되는
    argparse.Namespace 객체를 반환합니다.

    노트북에서 사용 예:
        from project_config import make_args
        args = make_args(num_steps=500, eval_every=50)
    """
    defaults = dict(
        name             = 'transfg_aircraft',
        dataset          = DATASET,
        data_root        = DATA_ROOT,
        model_type       = MODEL_TYPE,
        pretrained_dir   = PRETRAINED_WEIGHTS,
        pretrained_model = None,
        output_dir       = OUTPUT_DIR,
        img_size         = IMG_SIZE,
        train_batch_size = TRAIN_BATCH_SIZE,
        eval_batch_size  = EVAL_BATCH_SIZE,
        num_steps        = NUM_STEPS,
        warmup_steps     = WARMUP_STEPS,
        eval_every       = EVAL_EVERY,
        learning_rate    = LR,
        weight_decay     = WEIGHT_DECAY,
        max_grad_norm    = MAX_GRAD_NORM,
        decay_type       = DECAY_TYPE,
        smoothing_value  = SMOOTHING,
        split            = SPLIT,
        slide_step       = SLIDE_STEP,
        seed             = SEED,
        gradient_accumulation_steps = GRAD_ACCUM,
        fp16             = False,
        local_rank       = -1,      # 단일 디바이스: -1
        device           = DEVICE,
        n_gpu            = torch.cuda.device_count() if _dt == 'cuda' else 0,
        nprocs           = 1,
        pin_memory       = PIN_MEMORY,
        num_workers      = NUM_WORKERS,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


if __name__ == '__main__':
    print(f"PROJECT_ROOT       : {PROJECT_ROOT}")
    print(f"DATA_ROOT          : {DATA_ROOT}")
    print(f"  exists           : {os.path.isdir(DATA_ROOT)}")
    print(f"PRETRAINED_WEIGHTS : {PRETRAINED_WEIGHTS}")
    print(f"  ImageNet exists  : {os.path.isfile(IMAGENET_WEIGHTS)}")
    print(f"  CUB exists       : {os.path.isfile(CUB_PRETRAINED)}")
    print(f"DEVICE             : {DEVICE}")
    print(f"TRAIN_BATCH        : {TRAIN_BATCH_SIZE}")
    print(f"EVAL_BATCH         : {EVAL_BATCH_SIZE}")
    print(f"NUM_WORKERS        : {NUM_WORKERS}")
    print(f"PIN_MEMORY         : {PIN_MEMORY}")
