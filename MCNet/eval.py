#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from config import config
from furnace.utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from furnace.utils.visualize import print_iou, show_img
from furnace.engine.evaluator import Evaluator
from furnace.engine.logger import get_logger
from furnace.seg_opr.metric import hist_info, compute_score, pixelAccuracy
from furnace.datasets.ir_crack.IRT_Crack import IRT_Crack
from network import MCNet

import furnace.utils.misc2 as misc2

logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        #pred = self.sliding_eval(img, config.eval_crop_size,
                                 #config.eval_stride_rate, device)
        pred = self.whole_eval(img, None, device=device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes,
                                                       pred,
                                                       label)
        dice, specificity, precision, recall, accuracy, jaccard = self.evaluate_one_img(pred, label)
        results_dict = {
            'dice': dice,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'jaccard': jaccard,
        }
        if self.save_path is not None:
            fn = name + '_result.png'
            self.decode_segmap(pred, fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    # def compute_metric(self, results):
    #     hist = np.zeros((config.num_classes, config.num_classes))
    #     correct = 0
    #     labeled = 0
    #     count = 0
    #     for d in results:
    #         hist += d['hist']
    #         correct += d['correct']
    #         labeled += d['labeled']
    #         count += 1


    #     # pixel_accuracy, pixel_correct, pixel_labeled = pixelAccuracy(pred, label)
    #     iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
    #                                                    labeled)
    #     result_line = print_iou(iu, mean_pixel_acc,
    #                             dataset.get_class_names(), True)
    #     return result_line
    
    def evaluate_one_img(self, pred, gt):
        dice = misc2.dice(pred, gt)
        specificity = misc2.specificity(pred, gt)
        jaccard = misc2.jaccard(pred, gt)
        precision = misc2.precision(pred, gt)
        recall = misc2.recall(pred, gt)
        accuracy = misc2.accuracy(pred, gt)

        return dice, specificity, precision, recall, accuracy, jaccard

    # def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False,
    #           no_print=False):
    #     n = iu.size
    #     lines = []
    #     for i in range(n):
    #         if class_names is None:
    #             cls = 'Class %d:' % (i + 1)
    #         else:
    #             cls = '%d %s' % (i + 1, class_names[i])
    #         lines.append('%-8s\t%.3f%%' % (cls, iu[i] * 100))
    #     mean_IU = np.nanmean(iu)
    #     mean_IU_no_back = np.nanmean(iu[1:])
    #     if show_no_back:
    #         lines.append(
    #             '----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % (
    #                 'mean_IU', mean_IU * 100, 'mean_IU_no_back',
    #                 mean_IU_no_back * 100,
    #                 'mean_pixel_ACC', mean_pixel_acc * 100))
    #     else:
    #         print(mean_pixel_acc)
    #         lines.append(
    #             '----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%' % (
    #                 'mean_IU', mean_IU * 100, 'mean_pixel_ACC',
    #                 mean_pixel_acc * 100))
    #     line = "\n".join(lines)
    #     if not no_print:
    #         print(line)
    #     return line

    
    def compute_metric(self, results):
        dice_lst, specificity_lst, precision_lst, recall_lst, accuracy_lst, jaccard_lst = [], [], [], [], [], []
        for d in results:
            dice_lst.append(d['dice'])
            specificity_lst.append(d['specificity'])
            precision_lst.append(d['precision'])
            recall_lst.append(d['recall'])
            accuracy_lst.append(d['accuracy'])
            jaccard_lst.append(d['jaccard'])
        # mean
        dice = sum(dice_lst) / len(dice_lst)
        acc = sum(specificity_lst) / len(specificity_lst)
        precision = sum(precision_lst) / len(precision_lst)
        recall = sum(recall_lst) / len(recall_lst)
        accuracy = sum(accuracy_lst) / len(accuracy_lst)
        jac = sum(jaccard_lst) / len(jaccard_lst)
        result_line = 'Dice: %.4f, Specificity: %.4f, Precision: %.4f, Recall: %.4f, Accuracy: %.4f, Jaccard: %.4f' % (dice, acc, precision, recall, accuracy, jac)
        print(result_line)
        return result_line


    def decode_segmap(self, label_mask, fn):
        n_classes = config.num_classes
        label_colours = self.dataset.get_class_colors()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            # print(label_colours[ll, 0])
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        rgb = rgb*255
        cv2.imwrite(os.path.join(self.save_path, fn),  rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = MCNet(config.num_classes, criterion=None, edge_criterion=None)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    # 'eval_source': config.eval_source,
                    'test_source': config.test_source}
    dataset = IRT_Crack(data_setting, 'val', None)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
