"""
FGVC-Aircraft Dataset (2013b)
공식 배포: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

레이블 계층:
  variant     (100종) ← TransFG 논문에서 사용
  family      (70종)
  manufacturer(41종)

레이블 파일 형식: "<image_id> <class_name>"
이미지 파일:      data/images/<image_id>.jpg
"""
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class AircraftDataset(Dataset):
    """FGVC-Aircraft dataset.

    Args:
        root:  fgvc-aircraft-2013b 디렉토리 경로
        split: 'train' | 'val' | 'trainval' | 'test'
        level: 'variant' | 'family' | 'manufacturer'
        transform: torchvision transforms
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        level: str = "variant",
        transform=None,
    ):
        assert split in ("train", "val", "trainval", "test")
        assert level in ("variant", "family", "manufacturer")
        self.transform = transform

        data_dir = Path(root) / "data"
        label_file = data_dir / f"images_{level}_{split}.txt"
        assert label_file.exists(), f"레이블 파일 없음: {label_file}"

        # 클래스명 → index 매핑 (variants.txt / families.txt / manufacturers.txt 순서 기준)
        level_file_map = {
            "variant":      "variants.txt",
            "family":       "families.txt",
            "manufacturer": "manufacturers.txt",
        }
        class_list_file = data_dir / level_file_map[level]
        self._class_names = class_list_file.read_text().strip().splitlines()
        self._class_to_idx = {c: i for i, c in enumerate(self._class_names)}

        # 이미지 경로 및 레이블 파싱
        self.img_paths = []
        self.labels = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                img_id, class_name = parts[0], parts[1]
                self.img_paths.append(str(data_dir / "images" / f"{img_id}.jpg"))
                self.labels.append(self._class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        img = Image.open(self.img_paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def get_class_names(self) -> list[str]:
        return self._class_names
