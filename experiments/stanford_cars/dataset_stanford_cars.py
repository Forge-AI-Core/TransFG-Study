"""
Stanford Cars Dataset
HuggingFace tanganke/stanford_cars 에서 받은 parquet 형식을 처리.
이미지는 parquet 내부에 bytes로 내장되어 있음 (파일 시스템 접근 불필요).
"""
import io
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def _parse_class_names(readme_path: Path) -> list[str]:
    text = readme_path.read_text()
    pairs = re.findall(r"'(\d+)': (.+)", text)
    names = {int(k): v.strip().rstrip(",").strip("'\"") for k, v in pairs}
    return [names[i] for i in range(len(names))]


class StanfordCarsDataset(Dataset):
    """Stanford Cars (196 classes, parquet 형식).

    Args:
        root: data/Stanford_Cars 디렉토리 경로
        split: 'train' | 'test'
        transform: torchvision transforms
    """

    NUM_CLASSES = 196

    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "test"), "split은 'train' 또는 'test'"
        self.transform = transform

        data_dir = Path(root) / "data"
        parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
        assert parquet_files, f"parquet 파일 없음: {data_dir}/{split}-*.parquet"

        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        self._images_bytes = df["image"].tolist()   # list of dict {'bytes': ..., 'path': ...}
        self.labels = df["label"].tolist()

        readme = Path(root) / "README.md"
        self._class_names = _parse_class_names(readme) if readme.exists() else [str(i) for i in range(self.NUM_CLASSES)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        img_bytes = self._images_bytes[index]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def get_class_names(self) -> list[str]:
        return self._class_names
