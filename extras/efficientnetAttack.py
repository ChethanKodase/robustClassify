
import torch
from torchvision.models import efficientnet_v2_s

import numpy as np
import random

import matplotlib.pyplot as plt
from PIL import Image

from datasets import load_dataset

import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.paths = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # dummy label
        return img, -1


def main():
    set_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained on ImageNet
    model = efficientnet_v2_s(weights="IMAGENET1K_V1")
    model.to(device)
    model.eval()

    print("done")

    data_dir = "imagenetDataSubset"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = FlatImageDataset(data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Total images: {len(dataset)}")

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        print("labels", labels)
        print("images.shape", images.shape)
        break  # just first batch


if __name__ == "__main__":
    main()