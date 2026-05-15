"""
TransFG 추론(Inference) 유틸리티
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((600, 600), _BILINEAR),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict_single(
    model,
    image_path: str,
    device: torch.device,
    transform=None,
    class_names: list | None = None,
) -> dict:
    transform = transform or INFERENCE_TRANSFORM
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred = int(probs.argmax())
    confidence = float(probs[pred])

    result = {"pred_idx": pred, "confidence": confidence, "probs": probs}
    if class_names:
        result["pred_name"] = class_names[pred]
        top5_idx = probs.argsort()[::-1][:5]
        result["top5"] = [(class_names[i], float(probs[i])) for i in top5_idx]
    return result


@torch.no_grad()
def predict_batch(model, images: torch.Tensor, device: torch.device):
    model.eval()
    logits = model(images.to(device))
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs


@torch.no_grad()
def evaluate_dataset(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fct(logits, y).item()
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    acc = float((preds == labels).mean())
    return {"accuracy": acc, "avg_loss": total_loss / len(loader), "preds": preds, "labels": labels}
