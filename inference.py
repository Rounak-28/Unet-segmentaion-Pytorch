import torch
from torchvision.transforms import ToTensor, CenterCrop, Compose, RandomRotation, RandomResizedCrop, RandomCrop, ColorJitter, Normalize
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader


from variables import *
from model import UNet
from utils import show_predictions
from dataset import CustomDataset, test_dataloader
from augmentation import DoubleCompose, DoubleElasticTransform, DoubleHorizontalFlip, DoubleVerticalFlip, DoubleRandomResizedCrop, DoubleRandomRotation


model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load("models/model1.pth"))
model.to(device)



show_predictions(test_dataloader, model)