import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# -----------------------------
# 1. Device
# -----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std  = torch.tensor([0.229, 0.224, 0.225]).to(device)

criterion = nn.MSELoss()
desired_norm_l_inf = 0.1
#attackType = "untargeted"
attackType = "targeted"
def getPlot(images, name):
    img = images[0].permute(1, 2, 0)     # HWC
    img = img * std + mean               # Unnormalize
    img = img.clamp(0, 1)                # keep within [0,1]

    img_np = img.cpu().numpy()           # <<< MOVE TO CPU AND NUMPY

    plt.imshow(img_np)
    plt.axis('off')
    plt.savefig('plots/'+name+'.png')

def getPlot64(images, name):
    """
    images: Tensor [64, 3, 384, 384]
    Saves an 8x8 grid of unnormalized images.
    """
    images = images.detach().cpu()  # Move batch to CPU

    # Create figure
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    axes = axes.flatten()

    for i in range(64):
        img = images[i].permute(1, 2, 0)   # [H, W, C]

        # Unnormalize
        img = img * std.cpu() + mean.cpu()
        img = img.clamp(0, 1)

        axes[i].imshow(img.numpy())
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"plots/{name}.png")
    plt.close()


def main():
    set_seed(1)

    # -----------------------------
    # 2. Paths to ImageNet-Mini
    #    Adjust if your folder names differ
    # -----------------------------
    data_root = "/mdadm0/chethan_krishnamurth/robustClassify/imagenet-mini/imagenet-mini"
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")   # will use as "test"

    print("Train dir:", train_dir)
    print("Val/Test dir:", val_dir)

    # -----------------------------
    # 3. Transforms (match EfficientNetV2 pretrained)
    # -----------------------------
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    class_dict = weights.meta["categories"]

    preprocess = weights.transforms()  # includes resize, center crop, tensor, normalize

    # -----------------------------
    # 4. Datasets
    # -----------------------------
    train_dataset = ImageFolder(root=train_dir, transform=preprocess)
    test_dataset  = ImageFolder(root=val_dir,   transform=preprocess)

    print("Num train images:", len(train_dataset))
    print("Num test images:", len(test_dataset))
    print("Num classes:", len(train_dataset.classes))

    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = efficientnet_v2_s(weights=weights)
    model = model.to(device)
    model.eval()


    print("\n--- First TEST batch ---")
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        print("batch_idx:", batch_idx)
        print("labels:", labels)
        print("images.shape:", images.shape)
        break  
    
    originalClasses = [class_dict[top.item()] for top in labels]
    source_im = images

    mi, ma = source_im.min(), source_im.max()

    noise_addition = (torch.randn_like(images) * 0.2).cuda()
    noise_addition = noise_addition.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([noise_addition], lr=0.001)


    if attackType =="untargeted":
        for step in range(100):

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)
            logitsAttacked = model(normalized_attacked)       # [B, 1000]
            logitsNormal = model(source_im)       # [B, 1000]
            loss = criterion(logitsAttacked, logitsNormal)

            total_loss = -1 * loss 
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)


            if (step%10==0):
                with torch.no_grad():
                    print("step", step)
                    print("loss.item()", loss.item())
                    probsAttacked = torch.softmax(logitsAttacked, dim=1)
                    top1 = probsAttacked.argmax(dim=1)
                    samplesPreds = [class_dict[top.item()] for top in top1]
                    getPlot64(normalized_attacked, 'UntargtedAttacked')
                    getPlot64(source_im, 'original')
                    correct = (top1 == labels).sum().item()
                    Accuracy  = correct/len(labels)
                    print("Accuracy", Accuracy)
                    print()


    if attackType =="targeted":
        print("labels.shape", labels.shape)
        logitsNormal = torch.zeros(batch_size, 1000).to(device) # = model(source_im)       # [B, 1000]
        logitsNormal[:, 288]=1
        for step in range(500):
            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)
            logitsAttacked = model(normalized_attacked)       # [B, 1000]

            #print(logitsNormal.shape)
            #print("logitsNormal.shape", logitsNormal.shape)
            loss = criterion(logitsAttacked, logitsNormal)

            total_loss = loss 
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)


            if (step%10==0):
                with torch.no_grad():
                    print("step", step)
                    print("loss.item()", loss.item())
                    probsAttacked = torch.softmax(logitsAttacked, dim=1)
                    top1 = probsAttacked.argmax(dim=1)
                    samplesPreds = [class_dict[top.item()] for top in top1]
                    getPlot64(normalized_attacked, 'targetedAttacked')
                    getPlot64(source_im, 'original')
                    correct = (top1 == labels).sum().item()
                    Accuracy  = correct/len(labels)
                    print("Accuracy", Accuracy)
                    print()



        print("Original classes", originalClasses)
        print()
        print("samplesPreds", samplesPreds)

if __name__ == "__main__":
    main()