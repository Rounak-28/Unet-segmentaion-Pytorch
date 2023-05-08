import torch

from model import UNet
from variables import *
from utils import show_predictions
from dataset import test_dataloader

model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load("models/model1.pth"))
model.to(device)

show_predictions(test_dataloader, model)