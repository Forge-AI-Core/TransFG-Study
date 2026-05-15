"""
Stanford Cars용 DataLoader (TransFG 논문 설정 기준)
CUB data_loader.py와 동일한 인터페이스.
"""
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dataset_stanford_cars import StanfordCarsDataset

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR


def get_cars_transforms(img_size: int = 448):
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


def get_cars_loaders(
    data_root: str,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    img_size: int = 448,
    num_workers: int = 16,
):
    train_tf, test_tf = get_cars_transforms(img_size)
    trainset = StanfordCarsDataset(root=data_root, split="train", transform=train_tf)
    testset  = StanfordCarsDataset(root=data_root, split="test",  transform=test_tf)

    train_loader = DataLoader(
        trainset,
        sampler=RandomSampler(trainset),
        batch_size=train_batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        testset,
        sampler=SequentialSampler(testset),
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    return train_loader, test_loader, trainset, testset
