"""
TransFG Stanford Cars 학습 (Aircraft checkpoint 초기화)

처음 실행:
    TransFG_venv/bin/python run_cars_train.py --steps 10000

중간에 멈췄을 때 이어서:
    TransFG_venv/bin/python run_cars_train.py --steps 10000 --resume
"""
import sys, time, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--resume", action="store_true", help="checkpoint에서 이어서 학습")
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "TransFG"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from models.modeling import VisionTransformer, CONFIGS
from data_loader_cars import get_cars_loaders
from trainer import train

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT           = PROJECT_ROOT / "data/Stanford_Cars"
AIRCRAFT_CHECKPOINT = PROJECT_ROOT / "output/fgvc_aircraft_from_cub/transfg_aircraft_from_cub_checkpoint.bin"
OUTPUT_DIR          = PROJECT_ROOT / "output/stanford_cars_from_aircraft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_NAME            = "transfg_cars_from_aircraft"

print(f"Device              : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU                 : {torch.cuda.get_device_name(0)}")
print(f"DATA_ROOT           : {DATA_ROOT}")
print(f"OUTPUT_DIR          : {OUTPUT_DIR}")
print(f"steps               : {args.steps}")
print(f"resume              : {args.resume}")
print()

# 데이터로더 — num_workers=16
train_loader, test_loader, _, _ = get_cars_loaders(
    data_root        = str(DATA_ROOT),
    train_batch_size = 8,
    eval_batch_size  = 8,
    img_size         = 448,
    num_workers      = 16,
)
print(f"Train batches: {len(train_loader)}")
print(f"Test  batches: {len(test_loader)}\n")

# 모델 생성
config = CONFIGS["ViT-B_16"]
config.split      = "overlap"
config.slide_step = 12

model = VisionTransformer(
    config, img_size=448, zero_head=True, num_classes=196, smoothing_value=0.0,
)

if args.resume:
    # resume 모드: trainer.train() 내부에서 OUTPUT_DIR의 checkpoint를 자동 로드
    print("[Resume] trainer.train() 내부에서 checkpoint를 자동 로드합니다.")
else:
    # 처음 시작: Aircraft checkpoint에서 backbone만 로드 (head 제외)
    assert AIRCRAFT_CHECKPOINT.exists(), f"Aircraft checkpoint 없음: {AIRCRAFT_CHECKPOINT}"
    print(f"[처음 시작] Aircraft checkpoint에서 backbone 로드: {AIRCRAFT_CHECKPOINT}")
    ckpt = torch.load(str(AIRCRAFT_CHECKPOINT), map_location="cpu")
    aircraft_state_dict = ckpt["model"]
    # part_head (Aircraft 100 vs Cars 196) 제외
    filtered = {k: v for k, v in aircraft_state_dict.items() if not k.startswith("part_head")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  - 로드됨 : {len(filtered)} keys")
    print(f"  - missing (head): {missing}")
    print(f"  - Aircraft val_acc: {ckpt.get('best_acc', 0):.4f}")

model = model.to(DEVICE)
print(f"\n학습 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M\n")

# 학습 시작
print("=" * 60)
print(f"Stanford Cars training: {args.steps} steps  (resume={args.resume})")
print("=" * 60)
t0 = time.time()

best_acc, history = train(
    model        = model,
    train_loader = train_loader,
    test_loader  = test_loader,
    device       = DEVICE,
    num_steps    = args.steps,
    learning_rate= 3e-2,
    warmup_steps = 500 if args.steps >= 1000 else 50,
    decay_type   = "cosine",
    eval_every   = 100 if args.steps >= 1000 else 50,
    output_dir   = str(OUTPUT_DIR),
    run_name     = RUN_NAME,
    fp16         = True,
    gradient_accumulation_steps = 1,
    max_grad_norm = 1.0,
    resume       = args.resume,
)

elapsed = time.time() - t0
print(f"\n총 소요시간  : {elapsed:.1f}초 ({elapsed/3600:.2f}시간)")
print(f"step당 평균  : {elapsed/args.steps:.2f}초/step")
print(f"Best val acc : {best_acc:.4f} ({best_acc*100:.2f}%)")
