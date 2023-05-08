from variables import batch_size
from tifffile import imread
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import (
    Compose,
    ToTensor,
    ColorJitter,
    Normalize
)
from augmentation import (
    DoubleCompose,
    DoubleElasticTransform,
    DoubleHorizontalFlip,
    DoubleVerticalFlip,
    DoubleRandomResizedCrop,
    DoubleRandomRotation
)


class CustomDataset(Dataset):
    def __init__(self, img_transforms=None, mask_transforms=None, img_mask_transforms=None, test=False):
        self.images = imread("data/test-volume.tif") if test else imread("data/train-volume.tif")
        self.masks = imread("data/test-labels.tif") if test else imread("data/train-labels.tif")
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.img_mask_transforms = img_mask_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]  
        
        if self.img_transforms:
            image = self.img_transforms(image)
            
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        
        if self.img_mask_transforms:
            image, mask = self.img_mask_transforms(image, mask)

        return image, mask


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

training_data = CustomDataset(
    img_transforms=img_transforms, 
    mask_transforms=mask_transforms, 
    img_mask_transforms=img_mask_transforms, 
    test=False
)
testing_data = CustomDataset(
    img_transforms=img_transforms,
    mask_transforms=mask_transforms,
    img_mask_transforms=None,
    test=True
)


train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)