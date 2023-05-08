from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize
from torch.utils.data import Dataset
from tifffile import imread






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