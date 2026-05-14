"""
TransFG Single-GPU Trainer
apex 의존성 제거, torch.cuda.amp으로 FP16 대체.
분산 학습(distributed) 코드 제거 - 단일 GPU 학습 전용.
"""
import os
import math
import logging
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─── Schedulers (원본 utils/scheduler.py 인라인 복사) ──────────────────────────

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress)))


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


# ─── Utilities ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def save_checkpoint(output_dir: str, name: str, model, scaler=None,
                    optimizer=None, scheduler=None, global_step=0, best_acc=0.0):
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "model":       model.state_dict(),
        "global_step": global_step,
        "best_acc":    best_acc,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    path = os.path.join(output_dir, f"{name}_checkpoint.bin")
    torch.save(ckpt, path)
    logger.info(f"Checkpoint saved → {path}  (step={global_step})")
    return path


def load_checkpoint(path: str, model, device, scaler=None,
                    optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    global_step = ckpt.get("global_step", 0)
    best_acc    = ckpt.get("best_acc", 0.0)
    logger.info(f"Checkpoint loaded ← {path}  (step={global_step})")
    return global_step, best_acc


# ─── Validate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, test_loader, device):
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss()
    losses = AverageMeter()
    all_preds, all_labels = [], []

    for x, y in tqdm(test_loader, desc="Validating", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.update(loss_fct(logits, y).item())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = simple_accuracy(np.array(all_preds), np.array(all_labels))
    return losses.avg, acc


# ─── Train ─────────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    test_loader,
    device,
    num_steps: int = 10000,
    learning_rate: float = 3e-2,
    weight_decay: float = 0.0,
    warmup_steps: int = 500,
    decay_type: str = "cosine",
    eval_every: int = 100,
    output_dir: str = "./output",
    run_name: str = "transfg_cub",
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    resume: bool = False,   # True면 checkpoint에서 이어서 학습
):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", run_name))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    if decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps, num_steps)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps, num_steps)

    scaler = GradScaler() if fp16 else None

    global_step, best_acc = 0, 0.0

    # Resume from checkpoint
    ckpt_path = os.path.join(output_dir, f"{run_name}_checkpoint.bin")
    if resume and os.path.exists(ckpt_path):
        global_step, best_acc = load_checkpoint(
            ckpt_path, model, device, scaler, optimizer, scheduler
        )
        print(f"\n*** Resume from step {global_step} / {num_steps}  (best_acc={best_acc:.4f}) ***\n")
    elif resume:
        print(f"[경고] checkpoint 없음 ({ckpt_path}), 처음부터 학습합니다.")

    model.zero_grad()
    losses = AverageMeter()
    history = {"train_loss": [], "val_acc": [], "val_loss": []}

    while global_step < num_steps:
        model.train()
        all_preds, all_labels = [], []
        pbar = tqdm(train_loader, desc=f"[{global_step}/{num_steps}] loss=?")

        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            if fp16:
                with autocast():
                    loss, logits = model(x, y)
                loss = loss.mean() / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss, logits = model(x, y)
                loss = loss.mean() / gradient_accumulation_steps
                loss.backward()

            all_preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if (step + 1) % gradient_accumulation_steps == 0:
                effective_loss = loss.item() * gradient_accumulation_steps
                losses.update(effective_loss)
                history["train_loss"].append(effective_loss)

                if fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                writer.add_scalar("train/loss", effective_loss, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                pbar.set_description(f"[{global_step}/{num_steps}] loss={losses.val:.5f}")

                if global_step % eval_every == 0:
                    val_loss, val_acc = validate(model, test_loader, device)
                    history["val_acc"].append(val_acc)
                    history["val_loss"].append(val_loss)
                    writer.add_scalar("val/accuracy", val_acc, global_step)
                    writer.add_scalar("val/loss", val_loss, global_step)
                    print(f"\nStep {global_step:>5} | val_loss={val_loss:.5f} | val_acc={val_acc:.4f} | best={best_acc:.4f}")

                    if val_acc > best_acc:
                        save_checkpoint(output_dir, run_name, model, scaler,
                                        optimizer, scheduler, global_step, val_acc)
                        best_acc = val_acc

                    model.train()

                if global_step >= num_steps:
                    break

        train_acc = simple_accuracy(np.array(all_preds), np.array(all_labels))
        print(f"Epoch train_acc={train_acc:.4f}")
        losses.reset()

    writer.close()
    print(f"\n=== Training Complete ===")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_acc, history
