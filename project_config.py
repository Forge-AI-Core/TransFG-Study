# coding=utf-8
"""
project_config.py
TransFG 프로젝트 공통 설정
- 경로 자동 감지 (CUB 중첩 폴더 구조 처리)
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

# CUB-200-2011: tarball 해제시 중첩 폴더 구조 자동 감지
# /TransFG/CUB_200_2011/CUB_200_2011/ 또는 /TransFG/CUB_200_2011/
_cub_base = os.path.join(PROJECT_ROOT, 'CUB_200_2011')
DATA_ROOT = (
    os.path.join(_cub_base, 'CUB_200_2011')
    if os.path.isdir(os.path.join(_cub_base, 'CUB_200_2011'))
    else _cub_base
)

PRETRAINED_DIR = os.path.join(PROJECT_ROOT, 'pretrained', 'ViT-B_16.npz')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'output')
LOG_DIR        = os.path.join(PROJECT_ROOT, 'logs')


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
# macOS Jupyter 환경에서 num_workers > 0 시 deadlock 발생 가능 -> 0
NUM_WORKERS = 0 if platform.system() == 'Darwin' else 4
PIN_MEMORY  = (_dt == 'cuda')   # MPS/CPU에서는 pin_memory=False

# ── 모델 / 데이터 설정 ────────────────────────────────────────
DATASET       = 'CUB_200_2011'
NUM_CLASSES   = 200
MODEL_TYPE    = 'ViT-B_16'
IMG_SIZE      = 448
SPLIT         = 'non-overlap'   # 'non-overlap' | 'overlap'
SLIDE_STEP    = 12

# ── 학습 하이퍼파라미터 ───────────────────────────────────────
NUM_STEPS     = 10000
WARMUP_STEPS  = 500
EVAL_EVERY    = 100
LR            = 3e-2
WEIGHT_DECAY  = 0.0
MAX_GRAD_NORM = 1.0
DECAY_TYPE    = 'cosine'        # 'cosine' | 'linear'
SMOOTHING     = 0.0
SEED          = 42
GRAD_ACCUM    = 1


def make_args(**overrides) -> argparse.Namespace:
    """
    기존 utils/data_utils.py, models/modeling.py 등과 호환되는
    argparse.Namespace 객체를 반환합니다.

    노트북에서 사용 예:
        args = make_args(num_steps=500, eval_every=50)
        train_loader, test_loader = get_loader(args)
    """
    defaults = dict(
        name             = 'transfg_cub200',
        dataset          = DATASET,
        data_root        = DATA_ROOT,
        model_type       = MODEL_TYPE,
        pretrained_dir   = PRETRAINED_DIR,
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
        # DataLoader 추가 설정 (data_utils.py에서 getattr로 읽음)
        pin_memory       = PIN_MEMORY,
        num_workers      = NUM_WORKERS,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


if __name__ == '__main__':
    print(f"PROJECT_ROOT   : {PROJECT_ROOT}")
    print(f"DATA_ROOT      : {DATA_ROOT}")
    print(f"PRETRAINED_DIR : {PRETRAINED_DIR}")
    print(f"DEVICE         : {DEVICE}")
    print(f"TRAIN_BATCH    : {TRAIN_BATCH_SIZE}")
    print(f"EVAL_BATCH     : {EVAL_BATCH_SIZE}")
    print(f"NUM_WORKERS    : {NUM_WORKERS}")
    print(f"PIN_MEMORY     : {PIN_MEMORY}")
    print(f"DATA_ROOT exists: {os.path.isdir(DATA_ROOT)}")
    print(f"PRETRAINED exists: {os.path.isfile(PRETRAINED_DIR)}")
