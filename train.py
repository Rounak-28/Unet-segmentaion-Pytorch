import torch
import torch.nn as nn

from model import UNet
from variables import *
from utils import train, test
from dataset import train_dataloader, test_dataloader


model = UNet(in_channels=1, out_channels=1).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)



for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "models/model1.pth")