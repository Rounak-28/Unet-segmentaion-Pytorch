import random
import numpy as np

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from scipy.ndimage import map_coordinates, gaussian_filter
# from torchvision.transforms import RandomRotation, Compose


class DoubleCompose(T.Compose):

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class DoubleHorizontalFlip:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class DoubleVerticalFlip:
    """Apply vertical flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class DoubleElasticTransform:
    """Based on https://github.com/hayashimasa/UNet-PyTorch/blob/main/augmentation.py#L90-L142"""

    def __init__(self, alpha=250, sigma=10, p=0.5, seed=None, randinit=True):
        if not seed:
            seed = random.randint(1, 100)
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.randinit = randinit
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            if self.randinit:
                seed = random.randint(1, 100)
                self.random_state = np.random.RandomState(seed)
                self.alpha = random.uniform(100, 300)
                self.sigma = random.uniform(10, 15)
            
            dim = image.shape
            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            image = image.view(*dim[1:]).numpy()
            mask = mask.view(*dim[1:]).numpy()
            x, y = np.meshgrid(np.arange(dim[1]), np.arange(dim[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            image = map_coordinates(image, indices, order=1)
            mask = map_coordinates(mask, indices, order=1)
            image, mask = image.reshape(dim), mask.reshape(dim)
            image, mask = torch.Tensor(image), torch.Tensor(mask)

        return image, mask


class DoubleRandomRotation(T.RandomRotation):
    
    def __init__(self, degrees, expand=False, center=None):
        super(DoubleRandomRotation, self).__init__(degrees, expand, center)

    def __call__(self, img, mask):
        angle = self.get_params(self.degrees)
        img = TF.rotate(img, angle, False, self.expand, self.center)
        mask = TF.rotate(mask, angle, False, self.expand, self.center)

        return img, mask


class DoubleRandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
    
    def __call__(self, image, mask):
        _, width, height = image.size()
        area = width * height
        
        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))
            
            if random.random() < 0.5:
                w, h = h, w
            
            if w <= width and h <= height:
                left = random.randint(0, width - w)
                top = random.randint(0, height - h)
                
                image = TF.resized_crop(image, top, left, h, w, self.size, self.interpolation)
                mask = TF.resized_crop(mask, top, left, h, w, self.size, self.interpolation)
                
                return image, mask
        
        # If a crop could not be found after 10 attempts, fall back to center crop
        return TF.center_crop(image, self.size), TF.center_crop(mask, self.size)