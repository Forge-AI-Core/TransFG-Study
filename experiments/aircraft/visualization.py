"""
TransFG 시각화 유틸리티
- 샘플 이미지 그리드
- 학습 history 플롯
- 예측 결과 시각화
- Attention map 시각화 (forward hook 방식)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """ImageNet 정규화 역변환 → (H,W,3) float32 배열."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = (tensor.cpu() * std + mean).clamp(0, 1)
    return out.permute(1, 2, 0).numpy()


def show_sample_grid(dataset, class_names=None, n=12, cols=4, figsize=(14, 10), seed=42):
    """데이터셋에서 무작위 이미지 그리드 표시."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), n, replace=False)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        axes[i].imshow(denormalize(img))
        name = class_names[label] if class_names else f"Class {label}"
        axes[i].set_title(name.split(".")[-1][:22], fontsize=8)
        axes[i].axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_history(history: dict, eval_every: int = 100):
    """학습 loss / 검증 accuracy 플롯."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    if history.get("train_loss"):
        ax1.plot(history["train_loss"], linewidth=0.8)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.3)

    if history.get("val_acc"):
        steps = [(i + 1) * eval_every for i in range(len(history["val_acc"]))]
        ax2.plot(steps, history["val_acc"], marker="o", markersize=4)
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Accuracy")
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_predictions(images, preds, labels, class_names=None, cols=4):
    """배치 예측 결과 시각화 (초록=정답, 빨강=오답)."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(denormalize(images[i]))
        p, t = int(preds[i]), int(labels[i])
        pname = class_names[p].split(".")[-1] if class_names else str(p)
        tname = class_names[t].split(".")[-1] if class_names else str(t)
        color = "green" if p == t else "red"
        axes[i].set_title(f"P:{pname[:16]}\nT:{tname[:16]}", color=color, fontsize=7)
        axes[i].axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ─── Attention map 시각화 ──────────────────────────────────────────────────────

class AttentionMapExtractor:
    """
    TransFG Encoder의 self-attention weights를 forward hook으로 수집.
    model.transformer.encoder.layer[*].attn 에 hook을 등록한다.
    """

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self.attention_maps = []

    def __enter__(self):
        self.attention_maps = []
        for layer in self.model.transformer.encoder.layer:
            h = layer.attn.register_forward_hook(self._hook)
            self._hooks.append(h)
        return self

    def _hook(self, module, input, output):
        # Attention.forward returns (attn_output, weights)
        self.attention_maps.append(output[1].detach().cpu())

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def visualize_attention(model, img_tensor: torch.Tensor, device, head: int = 0, patch_size: int = 16, img_size: int = 448):
    """
    단일 이미지에 대한 attention map 시각화.
    img_tensor: (3, H, W) — 정규화된 텐서
    """
    x = img_tensor.unsqueeze(0).to(device)
    model.eval()

    with AttentionMapExtractor(model) as extractor:
        with torch.no_grad():
            _ = model(x)
        attn_maps = extractor.attention_maps  # list of (1, num_heads, seq_len, seq_len)

    if not attn_maps:
        print("Attention maps를 추출하지 못했습니다.")
        return

    n_layers = len(attn_maps)
    n_patches_side = img_size // patch_size  # 28 for 448/16

    fig, axes = plt.subplots(2, n_layers // 2 + n_layers % 2, figsize=(3 * (n_layers // 2 + 1), 6))
    axes = axes.flatten()

    for i, attn in enumerate(attn_maps):
        # attn: (1, num_heads, seq_len, seq_len)
        # CLS token → patch attention: [0, head, 0, 1:]
        cls_attn = attn[0, head, 0, 1:].numpy()  # (n_patches,)
        # non-overlap split: seq_len-1 == n_patches^2
        n_p = int(cls_attn.shape[0] ** 0.5)
        if n_p * n_p != cls_attn.shape[0]:
            axes[i].set_title(f"Layer {i+1}\n(overlap-skip)")
            axes[i].axis("off")
            continue
        attn_map = cls_attn.reshape(n_p, n_p)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        img_rgb = denormalize(img_tensor)
        attn_resized = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (img_size, img_size), Image.BILINEAR if not hasattr(Image, 'Resampling') else Image.Resampling.BILINEAR
        )) / 255.0

        axes[i].imshow(img_rgb)
        axes[i].imshow(attn_resized, alpha=0.5, cmap="jet")
        axes[i].set_title(f"Layer {i+1} Head {head}", fontsize=8)
        axes[i].axis("off")

    for ax in axes[n_layers:]:
        ax.axis("off")

    plt.suptitle("Self-Attention Maps (CLS→Patches)", fontsize=11)
    plt.tight_layout()
    plt.show()
