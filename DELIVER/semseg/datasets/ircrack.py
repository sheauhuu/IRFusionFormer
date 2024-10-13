import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
import sys

sys.path.append('path/to/your/DELIVER')
from semseg.augmentations_mm import get_train_augmentation

class IRCrack(Dataset):
    """
    num_classes: 2
    """
    CLASSES = ['background', 'crack']
    PALETTE = torch.tensor([[0,0,0],[255,255,255]])

    def __init__(self, root: str = 'path/to/your/Dataset', split: str = 'train', transform = None, modals = ['img', 'thermal'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = self._get_file_names(split)
    
        if not self.files:
            raise Exception(f"No images found in {self.root}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        rgb = os.path.join(*[self.root, '01_Visible_Image', os.path.basename(item_name)])
        x1 = os.path.join(*[self.root, '02_Infrared_Image', os.path.basename(item_name)])
        lbl_path = os.path.join(*[self.root, '04_Ground_Truth', os.path.basename(item_name).replace('png', 'jpg')])
        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if 'thermal' in self.modals:
            sample['thermal'] = self._open_img(x1)
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        label[label > 0] = 1
        sample['mask'] = label
        if self.transform is not None:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        return sample, label
        # return sample, label, item_name

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(self.root, '00_List/test.txt') if split_name == 'val' else os.path.join(self.root, '00_List/train_val.txt')
        file_names = []
        with open(source) as f:
            files = f.readlines()
        for item in files:
            file_name = item.strip()
            if ' ' in file_name:
                file_name = file_name.split(' ')[0].split('/')[-1]
            file_names.append(file_name)
        return file_names


if __name__ == '__main__':
    traintransform = get_train_augmentation((480, 480), seg_fill=255)

    trainset = IRCrack(transform=traintransform)
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))