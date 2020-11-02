import cv2
import numpy as np
import random


def extract_patches(image, size, stride):
    image_height, image_width, _ = image.shape
    
    patches = []
    for top in range(0, image_height - size[0] + 1, stride[0]):
        for left in range(0, image_width - size[1] + 1, stride[1]):
            patches.append(image[top:top + size[0], left:left + size[1]])
            
    return patches

def random_crop(image, size):
    image_height, image_width, _ = image.shape
    
    top = random.randint(0, image_height - size[0])
    left = random.randint(0, image_width - size[1])
    
    return image[top:top + size[0], left:left + size[1]]

class ForPatch:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, patches):
        return [self.transform(patch) for patch in patches]
    
class Albumentationed:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        augmented = self.transform(image=image)
        return augmented['image']

class GridCrop:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride
        
    def __call__(self, image):
        return extract_patches(image, self.size, self.stride)

class RandomNCrop:
    def __init__(self, size, number_of_patches):
        self.size = size
        self.number_of_patches = number_of_patches
        
    def __call__(self, image):
        patches = []
        for _ in range(0, self.number_of_patches):
            patches.append(random_crop(image, self.size))
        
        return patches

class RandomForegroundNCrop:
    def __init__(self, size, number_of_patches, number_of_attempts_per_patch, background_threshold=230, acceptable_background_ratio=0.5):
        self.size = size
        self.number_of_patches = number_of_patches
        self.number_of_attempts_per_patch = number_of_attempts_per_patch
        self.background_threshold = background_threshold
        self.acceptable_background_ratio = acceptable_background_ratio
    
    def __call__(self, image):
        patches = []
        for _ in range(0, self.number_of_patches):
            for _ in range(0, self.number_of_attempts_per_patch):
                patch = random_crop(image, self.size)
                if self._background_ratio_of(patch) < self.acceptable_background_ratio:
                    patches.append(patch)
                    break
            else:
                patches.append(random_crop(image, self.size))
            
        return patches
    
    def _background_ratio_of(self, patch):
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        is_background = patch > self.background_threshold
        return np.count_nonzero(is_background) / np.size(is_background)