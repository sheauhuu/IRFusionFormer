import numpy as np

from datasets.BaseDataset import BaseDataset
class IRT_Crack(BaseDataset):

    @classmethod
    def get_class_colors(*args):
        return np.array([
            [0, 0, 0],
            [255, 255, 255],
        ])

    @classmethod
    def get_class_names(*args):
        return ['background', 'crack']

    @classmethod
    def transform_label(cls, pred, name):
        pass