import os
import torch
import logging
import numpy as np
# import nibabel as nib
# import torchvision
import pickle
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pathlib
from tqdm import tqdm
from tqdm.contrib import tzip
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
# from monai import transforms
from albumentations import PadIfNeeded, OneOf, ElasticTransform, GridDistortion, OpticalDistortion, RandomRotate90, GaussNoise, RandomFog, RandomGamma, Normalize, Resize, ToGray, Compose, RandomSizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, ShiftScaleRotate, ToFloat
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
import matplotlib.pyplot as plt

class IRCrackDataset(Dataset):
    def __init__(self, list_path: pathlib.Path = None,rgb_path: pathlib.Path = None, 
                 infrared_path: pathlib.Path = None, fusion_path: pathlib.Path = None, 
                 gt_path: pathlib.Path = None, size: Tuple[int, int] = (480, 640), 
                 transform=True, split=None) -> None:
        super().__init__()
        self.list_path      = list_path
        self.rgb_path       = rgb_path
        self.infrared_path  = infrared_path
        self.gt_path        = gt_path
        file_name = []
        self.size = size
        self.transform = transform
        self.split = split
        self.data = []
        T = torch.from_numpy
        if self.split == 'train':
            # read txt file
            with open(pathlib.Path(self.list_path) / 'train_val.txt', 'r') as f:
                self.files = [name.strip().split(' ')[0].split('/')[-1] for idx, name in enumerate(f)]
        elif self.split == 'val':
            with open(pathlib.Path(self.list_path) / 'test.txt', 'r') as f:
                self.files = [name.strip().split(' ')[0].split('/')[-1] for idx, name in enumerate(f)]

        for file in tqdm(self.files): ######
            rgb_file_path = pathlib.Path(self.rgb_path) / file
            infrared_file_path = pathlib.Path(self.infrared_path) / file
            gt_file_path = pathlib.Path(self.gt_path) / file.replace('.png', '.jpg')
            # gt = np.array(Image.open(gt_file_path)).astype(np.uint8)
            # gt[gt > 0] = 1
            self.data.append({
                'name': file,
                # 'rgb': np.array(Image.open(rgb_file_path).convert('RGB')).astype(np.int8),
                # 'infrared': np.array(Image.open(infrared_file_path).convert('L')).astype(np.int8),
                # 'gt': np.array(Image.open(gt_file_path).convert('1')).astype(np.int8),
                'rgb': np.array(Image.open(rgb_file_path).convert('RGB')).astype(np.uint8),
                'infrared': np.array(Image.open(infrared_file_path).convert('L')).astype(np.uint8),
                'gt': np.array(Image.open(gt_file_path).convert('1')).astype(np.uint8),
                # 'gt': gt,
            })
        if transform == True:
            if self.split == 'train':
                self.transform_spatial = Compose([
                    RandomRotate90(p=0.6),
                    PadIfNeeded(min_height=self.size[0], min_width=self.size[1], always_apply=True),
                    # RandomSizedCrop(min_max_height=(300, 440), height=400, width=400, p=0.8),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=10, p=0.2),
                    OneOf([
                        ElasticTransform(alpha=1, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.01),
                        GridDistortion(p=1.0),
                        OpticalDistortion(distort_limit=0.03, shift_limit=0.03, p=0.1)                  
                    ], p=0.2), #弹性变化、网格畸变、光学畸变
                    # Resize(self.size[0], self.size[1], always_apply=True),])
            ], additional_targets={'infrared': 'image'})
                self.transform_color = Compose([
                    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.01),
                    RandomGamma(p=0.03),
                    GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=1.0),
                    RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=0.1),
                ])
                self.norm_rgb = Compose([
                    Resize(self.size[0], self.size[1], always_apply=True),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True),
                ])
                self.norm_infrared = Compose([
                    Resize(self.size[0], self.size[1], always_apply=True),
                    # infrared [0,1]
                    Normalize(mean=0.0, std=1.0, max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True),
                ])
            else:
                self.transform_spatial = Compose([
                    Resize(self.size[0], self.size[1], always_apply=True),], additional_targets={'infrared': 'image'})
                self.transform_color = Compose([])
                self.norm_rgb = Compose([
                    Resize(self.size[0], self.size[1], always_apply=True),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True),
                ])
                self.norm_infrared = Compose([
                    Resize(self.size[0], self.size[1], always_apply=True),
                    # infrared [0,1]
                    Normalize(mean=0.0, std=1.0, max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True),
                ])
        elif transform == False:
            self.transform_spatial = Compose([
                Resize(self.size[0], self.size[1], always_apply=True),], additional_targets={'infrared': 'image'})
            self.transform_color = Compose([])
            self.norm_rgb = Compose([
                Resize(self.size[0], self.size[1], always_apply=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True),
                ToTensorV2(always_apply=True),
            ])
            self.norm_infrared = Compose([
                Resize(self.size[0], self.size[1], always_apply=True),
                # infrared [0,1]
                Normalize(mean=0.0, std=1.0, max_pixel_value=255.0, always_apply=True),
                ToTensorV2(always_apply=True),
            ])
        else:
            self.transform_spatial = transform['spatial']
            self.transform_color = transform['color']
            self.norm_rgb = transform['norm_rgb']
            self.norm_infrared = transform['norm_infrared']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img_name, img_rgb, img_infrared, img_gt = data['name'], data['rgb'], data['infrared'], data['gt']
        img_infrared = np.expand_dims(img_infrared, axis=2)
        img_gt = np.expand_dims(img_gt, axis=2)
        assert img_rgb.shape[:2] == img_infrared.shape[:2] == img_gt.shape[:2]
        # img_temp = np.concatenate([img_rgb, img_infrared], axis=2)
        # img_spatial_temp = self.transform_spatial(image=img_temp, mask=img_gt)
        # img_rgb = img_spatial_temp['image'][:, :, :3].astype(np.float32)
        # img_infrared = img_spatial_temp['image'][:, :, 3:].astype(np.float32)
        # img_gt = img_spatial_temp['mask']
        img_spatial = self.transform_spatial(image=img_rgb, infrared=img_infrared, mask=img_gt)
        img_rgb = img_spatial['image']
        img_infrared = img_spatial['infrared']
        img_gt = img_spatial['mask']
        img_rgb = self.transform_color(image=img_rgb)['image']
        img_norm_temp = self.norm_rgb(image=img_rgb, mask=img_gt)
        img_rgb = img_norm_temp['image']
        img_gt = img_norm_temp['mask']
        img_infrared = self.norm_infrared(image=img_infrared)['image']
        img_gt = img_gt.squeeze(2).unsqueeze(0)
        return {
            'name': img_name,
            'rgb': img_rgb,
            'infrared': img_infrared,
            'gt': img_gt,
        }


def get_crack_dataset(configs, split='train'):
    assert split in ['train', 'val'], 'split should be either train or val'
    list_root_path = configs['list_root']
    rgb_root_path = configs['rgb_root']
    gt_root_path = configs['gt_root']
    infrared_root_path = configs['infrared_root']
    img_size   = configs['img_size']
    val_batch_size = configs['val_batch_size']
    train_batch_size = configs['train_batch_size']
    batch_size = {"train": train_batch_size, "val": val_batch_size}[split]
    split = split
    if 'transform' in configs.keys():
        transform = configs['transform']
    else:
        transform = True

    assert os.path.exists(rgb_root_path), 'rgb_root_path does not exist'
    assert os.path.exists(gt_root_path), 'gt_root_path does not exist'
    assert os.path.exists(infrared_root_path), 'infrared_root_path {} does not exist'.format(infrared_root_path)

    crack_dataset = IRCrackDataset(list_path=list_root_path,rgb_path=rgb_root_path, infrared_path=infrared_root_path, gt_path=gt_root_path, size=img_size, split=split, transform=transform)
    if split == 'train':
        data_loader = DataLoader(crack_dataset, batch_size=batch_size, num_workers=configs['num_workers'], shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=lambda _: np.random.seed())
    else:
        data_loader = DataLoader(crack_dataset, batch_size=batch_size, num_workers=configs['num_workers'], shuffle=False, drop_last=False, pin_memory=True, worker_init_fn=lambda _: np.random.seed())
    return data_loader

def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    # image = np.clip(image, 0, 1)
    return image

def visualize_image(data):
    # 处理RGB图像
    rgb = data['rgb'][0].permute(1, 2, 0).numpy()
    rgb = denormalize_image(rgb)
    # 处理红外图像
    infrared = data['infrared'][0].permute(1, 2, 0).numpy()
    # min and max in the infrared image
    print(infrared.min(), infrared.max())
    # print(data['infrared'][0])
    infrared = infrared.squeeze()
    # 处理Ground truth图像
    gt = data['gt'][0].squeeze().numpy()

    plt.figure(figsize=(12, 8))
    # RGB图像
    plt.subplot(2, 3, 1)
    plt.imshow(rgb)
    plt.title('RGB Image')
    # 红外图像
    plt.subplot(2, 3, 2)
    plt.imshow(infrared, cmap='gray')
    plt.title('Infrared Image')
    # Ground truth图像
    plt.subplot(2, 3, 3)
    plt.imshow(gt, cmap='gray')
    plt.title('Ground Truth Image')

    rgb_original = np.array(Image.open('path/to/your/Dataset/01_Visible_Image/' + data['name'][0]).convert('RGB'))
    infrared_original = np.array(Image.open('path/to/your/Dataset/02_Infrared_Image/' + data['name'][0]).convert('L'))
    gt_original = np.array(Image.open('path/to/your/Dataset/04_Ground_Truth/' + data['name'][0].replace('.png','.jpg')).convert('1'))
    
    # RGB图像
    plt.subplot(2, 3, 4)
    plt.imshow(rgb_original)
    plt.title('Original RGB Image')
    # 红外图像
    plt.subplot(2, 3, 5)
    plt.imshow(infrared_original, cmap='gray')
    plt.title('Original Infrared Image')
    # Ground truth图像
    plt.subplot(2, 3, 6)
    plt.imshow(gt_original, cmap='gray')
    plt.title('Original Ground Truth Image')
    
    plt.tight_layout()
    # plt.show()
    # plt.savefig('test.png')

if __name__ == '__main__':
    # test the dataset
    configs = {
        'list_root': 'path/to/your/Dataset/00_List',
        'rgb_root': 'path/to/your/Dataset/01_Visible_Image',
        'gt_root': 'path/to/your/Dataset/04_Ground_Truth',
        'infrared_root': 'path/to/your/Datasetd/02_Infrared_Image',
        'img_size': (480, 480),
        'val_batch_size': 1,
        'train_batch_size': 1,
        'num_workers': 1,
    }
    data_loader = get_crack_dataset(configs, split='val')
    for data in data_loader:
        print(data['name'])
        print(data['rgb'].shape)
        print(data['infrared'].shape)
        print(data['gt'].shape)
        break

    # visualize the dataset
    for data in data_loader:
        visualize_image(data)
        break
    '''
        ['LAB00229.png']
        torch.Size([1, 3, 480, 640])
        torch.Size([1, 1, 480, 640])
        torch.Size([1, 1, 480, 640])
    '''