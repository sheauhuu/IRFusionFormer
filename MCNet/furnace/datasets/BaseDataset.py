#!/usr/bin/env python3
# encoding: utf-8
# @Author  : xionghaitao
# @File    : BaseDataset.py

import os
import time
import cv2
import torch
import numpy as np

import torch.utils.data as data
from PIL import Image

class BaseDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None):
        super(BaseDataset, self).__init__()
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        # self._eval_source = setting['eval_source']
        self._test_source = setting['test_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.valid_classes = [0, 10]
        self.class_map = dict(zip(self.valid_classes, range(2)))

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = os.path.join(self._img_path, names[0])
        gt_path = os.path.join(self._gt_path, names[1])
        item_name = names[1].split("/")[-1].split(".")[0]

        img, gt = self._fetch_data(img_path, gt_path)
        img = img[:, :, ::-1]
        if self._split_name == 'val':
            gt = self.encode_segmap(np.ascontiguousarray(gt))
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        if self._split_name == 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(self.encode_segmap(np.ascontiguousarray(gt))).long()
            # img = torch.from_numpy(np.ascontiguousarray(img)).float()
            # gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name),
                           n=len(self._file_names))
        # output_dict = dict(data=img, label=gt)
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        return img, gt

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']

        file_names = []
        if split_name == 'train':
            source = self._train_source
            with open(source) as f:
                files = f.readlines()
        else:
            source = self._test_source
            # source = [path1, path2]
            files = []
            with open(source) as f:
                files += f.readlines()

        for item in files:
            img_name, gt_name = self._process_item_names(item)
            file_names.append([img_name, gt_name])

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)
        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]  # 加上余下的

        return new_file_names

    @staticmethod
    def _process_item_names(item):
        item = item.strip().split(' ')
        img_name = item[1].split('/')[-1]
        gt_name = item[-1].split('/')[-1]

        return img_name, gt_name

    def get_length(self):
        return self.__len__()
    
    def get_sample(self, index):
        return self.__getitem__(index)

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        if mode == cv2.IMREAD_GRAYSCALE:
            # repalce all the non-zero values to 200
            img[img > 0] = 10
        return img

    @classmethod
    def get_class_colors(*args):
        raise NotImplementedError

    @classmethod
    def get_class_names(*args):
        raise NotImplementedError

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

if __name__ == "__main__":
    data_setting = {'img_root': 'path/to/your/Dataset/02_Infrared_Image',
                    'gt_root': 'path/to/your/Dataset/04_Ground_Truth',
                    'train_source': 'path/to/your/Dataset/00_List/train_val.txt',
                    'test_source': 'path/to/your/Dataset/00_List/test.txt'}
    bd = BaseDataset(data_setting, 'val', None)
    print(bd.get_length())
    print(bd.get_sample(10))
    # print(bd.get_class_names())
