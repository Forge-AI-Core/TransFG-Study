"""
FGVC-Aircraft용 DataLoader (TransFG 논문 설정 기준)
CUB data_loader.py와 동일한 인터페이스.
"""
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dataset_aircraft import AircraftDataset

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR


def get_aircraft_transforms(img_size: int = 448):
    train = transforms.Compose([
        transforms.Resize((600, 600), _BILINEAR),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test = transforms.Compose([
        transforms.Resize((600, 600), _BILINEAR),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, test


def get_aircraft_loaders(
    data_root: str,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    img_size: int = 448,
    num_workers: int = 4,
    level: str = "variant",        # TransFG 논문: variant (100클래스)
    use_trainval: bool = False,    # True면 train+val 합쳐서 학습
):
    train_tf, test_tf = get_aircraft_transforms(img_size)

    train_split = "trainval" if use_trainval else "train"
    trainset = AircraftDataset(root=data_root, split=train_split, level=level, transform=train_tf)
    valset   = AircraftDataset(root=data_root, split="val",       level=level, transform=test_tf)
    testset  = AircraftDataset(root=data_root, split="test",      level=level, transform=test_tf)

    train_loader = DataLoader(
        trainset,
        sampler=RandomSampler(trainset),
        batch_size=train_batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valset,
        sampler=SequentialSampler(valset),
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        sampler=SequentialSampler(testset),
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, trainset, valset, testset
