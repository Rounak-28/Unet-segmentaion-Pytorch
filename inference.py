import torch
from torchvision.transforms import ToTensor, CenterCrop, Compose, RandomRotation, RandomResizedCrop, RandomCrop, ColorJitter, Normalize
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader


from model import UNet
from utils import show_predictions
from dataset import CustomDataset
from augmentation import DoubleCompose, DoubleElasticTransform, DoubleHorizontalFlip, DoubleVerticalFlip, DoubleRandomResizedCrop, DoubleRandomRotation

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load("models/model1.pth"))
model.to(device)

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

testing_data = CustomDataset(img_transforms=img_transforms, mask_transforms=mask_transforms, img_mask_transforms=None, test=True)
test_dataloader = DataLoader(testing_data, batch_size=2)

show_predictions(test_dataloader, model, device)