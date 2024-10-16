import cv2
import torch
import numpy as np
from torch.utils import data

from config import config
from furnace.utils.img_utils import random_scale, random_mirror, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape, contrast_and_Brightness
import matplotlib.pyplot as plt

class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std
        edge_radius = 7
        self.edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (edge_radius, edge_radius))

    def __call__(self, img, gt):
        img = contrast_and_Brightness(img)
        img, gt = random_mirror(img, gt)
        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)

        id255 = np.where(gt == 255)
        no255_gt = np.array(gt)
        no255_gt[id255] = 0
        # check no255_gt is all zero    
        if np.sum(no255_gt) == 0:
            print('no255_gt is all zero')
            plt.imshow(gt)
            plt.show()
        cgt = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
        cgt = cv2.dilate(cgt, self.edge_kernel)
        cgt[cgt == 255] = 1
        img = normalize(img, self.img_mean, self.img_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_cgt, _ = random_crop_pad_to_shape(cgt, crop_pos, crop_size, 255)

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = {'aux_label': p_cgt}

        return p_img, p_gt, extra_dict



def get_train_loader(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    # 'eval_source': config.eval_source,
                    'test_source': config.test_source}
    train_preprocess = TrainPre(config.image_mean, config.image_std)

    # train_dataset = dataset(data_setting, "train", train_preprocess,
    #                         config.niters_per_epoch * config.batch_size)
    train_dataset = dataset(data_setting, "train", train_preprocess)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    # if engine.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_dataset)
    #     batch_size = config.batch_size // engine.world_size
    #     is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

def get_val_loader(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    # 'eval_source': config.eval_source,
                    'test_source': config.test_source}
    train_preprocess = TrainPre(config.image_mean, config.image_std)

    # train_dataset = dataset(data_setting, "train", train_preprocess,
    #                         config.niters_per_epoch * config.batch_size)
    train_dataset = dataset(data_setting, "train", train_preprocess)

    train_sampler = None
    is_shuffle = False
    batch_size = 1

    # if engine.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_dataset)
    #     batch_size = config.batch_size // engine.world_size
    #     is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

if __name__ == '__main__':
    from config import config
    from torch.utils.data import DataLoader
    from furnace.datasets.ir_crack.IRT_Crack import IRT_Crack

    train_loader, train_sampler = get_train_loader(None, IRT_Crack)
    # for i, (img, gt, extra_dict) in enumerate(train_loader):
    #     print(i, (img, gt, extra_dict))
    #     print(gt)
    #     print(img.shape, gt.shape)
    #     print(extra_dict['aux_label'].shape)
    #     break
    for i, batch in enumerate(train_loader):
        img = batch['data']
        gt = batch['label']
        extra_dict = {key: value for key, value in batch.items() if key not in ['data', 'label']}
        print(i, (img, gt, extra_dict))
        print(img.shape, gt.shape)
        break
    print('Done')