# coding=utf-8
"""
train_utils.py
TransFG 학습/검증 유틸리티

원본 train.py 대비 변경 사항:
  - apex (FP16) 제거  ->  CUDA: torch.amp.GradScaler, MPS/CPU: FP32
  - torch.distributed 제거  ->  단일 디바이스 학습
  - CUDA / MPS / CPU 모두 지원
  - scipy.misc.imread 의존성 없음
"""
from __future__ import absolute_import, division, print_function

import os
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)


# ── 유틸리티 ──────────────────────────────────────────────────

class AverageMeter:
    """평균 및 현재 값 추적"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val    = float(val)
        self.sum   += float(val) * n
        self.count += n
        self.avg    = self.sum / self.count


def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def set_seed(seed: int, device_type: str = 'cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed_all(seed)


# ── 체크포인트 저장 / 불러오기 ────────────────────────────────

def save_checkpoint(output_dir: str, name: str, model: nn.Module, scaler=None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}_checkpoint.bin")
    ckpt = {'model': model.state_dict()}
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()
    torch.save(ckpt, path)
    logger.info("체크포인트 저장: %s", path)
    return path


def load_checkpoint(path: str, model: nn.Module, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    logger.info("체크포인트 로드: %s", path)
    return model


# ── 모델 초기화 ───────────────────────────────────────────────

def setup_model(args) -> nn.Module:
    """
    VisionTransformer 초기화 후 Google 사전학습 가중치(ViT-B_16.npz) 로드.
    args.pretrained_model 이 지정된 경우 파인튜닝 가중치도 로드.
    """
    config             = CONFIGS[args.model_type]
    config.split       = args.split
    config.slide_step  = args.slide_step

    _nc_map = {
        'CUB_200_2011': 200, 'car': 196, 'nabirds': 555,
        'dog': 120, 'INat2017': 5089,
    }
    num_classes = _nc_map[args.dataset]

    model = VisionTransformer(
        config,
        args.img_size,
        zero_head=True,
        num_classes=num_classes,
        smoothing_value=args.smoothing_value,
    )

    # Google 사전학습 가중치 (ImageNet-21k)
    model.load_from(np.load(args.pretrained_dir))
    logger.info("사전학습 가중치 로드 완료: %s", args.pretrained_dir)

    # 파인튜닝 체크포인트 (선택)
    if args.pretrained_model is not None:
        ft_weights = torch.load(args.pretrained_model, map_location='cpu')['model']
        model.load_state_dict(ft_weights)
        logger.info("파인튜닝 가중치 로드: %s", args.pretrained_model)

    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("총 학습 가능 파라미터: %.1fM", n_params / 1e6)
    return model


# ── 검증 ─────────────────────────────────────────────────────

def validate(model: nn.Module, test_loader, device: torch.device):
    """
    검증 루프 (단일 디바이스).
    반환: (accuracy: float, avg_loss: float)
    """
    model.eval()
    loss_fn  = nn.CrossEntropyLoss()
    avg_loss = AverageMeter()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Validation', leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)                   # labels 없이 호출 -> logits만 반환
            loss   = loss_fn(logits, y)
            avg_loss.update(loss.item(), n=x.size(0))
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy   = simple_accuracy(all_preds, all_labels)
    return accuracy, avg_loss.avg


# ── 학습 ─────────────────────────────────────────────────────

def train(args, model: nn.Module, train_loader, test_loader) -> float:
    """
    학습 루프 (단일 디바이스, apex 없음).

    디바이스별 AMP 전략:
      CUDA : torch.cuda.amp.GradScaler + autocast (FP16)
      MPS  : FP32 (MPS에서 float16 AMP 미지원)
      CPU  : FP32

    반환: best_val_accuracy (float)
    """
    device  = args.device
    use_amp = (device.type == 'cuda')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    writer  = SummaryWriter(log_dir=os.path.join('logs', args.name))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    t_total   = args.num_steps
    scheduler = (
        WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        if args.decay_type == 'cosine'
        else WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    )

    # AMP scaler: CUDA 전용
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    global_step = 0
    best_acc    = 0.0
    losses      = AverageMeter()
    start_time  = time.time()

    logger.info("=" * 60)
    logger.info("학습 시작")
    logger.info("  device    : %s", device)
    logger.info("  AMP(FP16) : %s", use_amp)
    logger.info("  total steps: %d", t_total)
    logger.info("=" * 60)

    model.zero_grad()
    model.train()

    while True:
        all_preds_ep, all_labels_ep = [], []

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Train {global_step}/{t_total}", leave=False)
        ):
            x, y = x.to(device), y.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, logits = model(x, y)
                loss = loss.mean() / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss, logits = model(x, y)
                loss = loss.mean() / args.gradient_accumulation_steps
                loss.backward()

            preds = torch.argmax(logits.detach(), dim=-1)
            all_preds_ep.append(preds.cpu().numpy())
            all_labels_ep.append(y.cpu().numpy())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("train/loss", losses.val,  global_step)
                writer.add_scalar("train/lr",   current_lr,  global_step)

                if global_step % args.eval_every == 0:
                    val_acc, val_loss = validate(model, test_loader, device)
                    logger.info(
                        "[Step %5d/%d] loss=%.4f | val_acc=%.4f | val_loss=%.4f | best=%.4f",
                        global_step, t_total, losses.avg, val_acc, val_loss, best_acc,
                    )
                    writer.add_scalar("val/accuracy", val_acc,  global_step)
                    writer.add_scalar("val/loss",     val_loss, global_step)

                    if val_acc > best_acc:
                        best_acc = val_acc
                        save_checkpoint(args.output_dir, args.name, model, scaler)
                        logger.info("  -> 최고 정확도 갱신: %.4f (저장 완료)", best_acc)
                    model.train()

                if global_step >= t_total:
                    break

        if global_step >= t_total:
            break

        ep_acc = simple_accuracy(
            np.concatenate(all_preds_ep), np.concatenate(all_labels_ep)
        )
        losses.reset()
        logger.info("  epoch 완료 | train_acc=%.4f | step=%d", ep_acc, global_step)

    writer.close()
    elapsed = (time.time() - start_time) / 3600
    logger.info("=" * 60)
    logger.info("학습 완료 | Best Val Acc: %.4f | 소요 시간: %.2fh", best_acc, elapsed)
    logger.info("=" * 60)
    return best_acc
