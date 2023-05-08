import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, CenterCrop, Compose, RandomRotation, RandomResizedCrop, RandomCrop, ColorJitter, Normalize

import random
import zipfile
from PIL import Image
from torchmetrics import JaccardIndex

from dataset import CustomDataset
from augmentation import DoubleCompose, DoubleElasticTransform, DoubleHorizontalFlip, DoubleVerticalFlip, DoubleRandomResizedCrop, DoubleRandomRotation
from utils import visualize_imgs, show_predictions
from model import UNet

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
img_size = 512

img_mask_transforms = DoubleCompose([
    DoubleElasticTransform(alpha=250, sigma=10),
    DoubleHorizontalFlip(),
    DoubleVerticalFlip(),
    # DoubleRandomResizedCrop(size=(512, 512)),
    # DoubleRandomRotation(degrees=(0, 180)),
])

img_transforms = Compose([
    ToTensor(),
    ColorJitter(brightness=.4),
    Normalize(0.5347, 0.2255),
])

mask_transforms = Compose([
    ToTensor()
])


training_data = CustomDataset(img_transforms=img_transforms, mask_transforms=mask_transforms, img_mask_transforms=img_mask_transforms, test=False)
testing_data = CustomDataset(img_transforms=img_transforms, mask_transforms=mask_transforms, img_mask_transforms=None, test=True)


train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)


# visualize_imgs(train_dataloader, is_mask=False)
# visualize_imgs(train_dataloader, is_mask=True)



# for x, y in train_dataloader:
#     print(x.shape)
#     print(y.shape)
#     break

model = UNet(in_channels=1, out_channels=1).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

train_loss_track = []
test_loss_track = []
test_iou_track = []

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # loss tracking
        train_loss_track.append(loss.item())

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.train()
    # IOU score i.e intersetion over union, ranges between 0-1, the higher the better
    test_loss, iou_score = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            jaccard = JaccardIndex(task="binary").to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            iou = jaccard(pred,y)
            iou_score += iou.item()
            test_loss += loss.item()
            # loss tracking
            test_loss_track.append(loss.item())
            test_iou_track.append(iou.item())

    iou_score /= num_batches
    test_loss /= num_batches
    print(f"Test loss: {test_loss:.4f}, IOU score: {iou_score:.4f}")


epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "models/model1.pth")