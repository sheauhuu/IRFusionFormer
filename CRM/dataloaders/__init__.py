# Written by Ukcheol Shin, Jan. 24, 2023
# Email: shinwc159@gmail.com
from .IRC_dataset import IRC_dataset
from .augmentation import *

def build_dataset(cfg):
    """
    Return corresponding dataset according to given dataset option
    :param config option
    :return Dataset class
    """

    # Set dataset
    dataset_name = cfg.DATASETS.NAME
    dataset={}
    if dataset_name == 'IRCdataset': # PST900 dataset doesn't has validation set
        dataset['train'] = IRC_dataset(cfg.DATASETS.DIR, cfg, split='train')
        dataset['val']   = IRC_dataset(cfg.DATASETS.DIR, cfg, split='test')
        dataset['test']  = IRC_dataset(cfg.DATASETS.DIR, cfg, split='test')
    else:
        raise ValueError('Unknown dataset type: {}.'.format(dataset_name))

    return dataset
