from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
from tools import misc2
from tools.metrics import Metrics

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def evaluate_one_img(pred, gt):
    dice = misc2.dice(pred, gt)
    specificity = misc2.specificity(pred, gt)
    jaccard = misc2.jaccard(pred, gt)
    precision = misc2.precision(pred, gt)
    recall = misc2.recall(pred, gt)
    accuracy = misc2.accuracy(pred, gt)

    return dice, specificity, precision, recall, accuracy, jaccard

def test(test_data_path='data/test_example.txt',
         save_path='deepcrack_results/',
         pretrained_model='checkpoints/DeepCrack_CT260_FT1.pth', ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    test_pipline = dataReadPip(transforms=None)

    test_list = readIndex(test_data_path)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)
    
    # model.load_state_dict(torch.load(pretrained_model))
    model_weight = torch.load(pretrained_model)
    from collections import OrderedDict
    multi_gpu_obj = OrderedDict()
    for k, v in model_weight.items():
        name = 'module.' + k  # add `module.`
        multi_gpu_obj[name] = v
    model_weight = multi_gpu_obj
    model.load_state_dict(model_weight)
    # model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=False))

    model.eval()

    metrics = Metrics(2, 255, 'cpu')

    with torch.no_grad():
        list_dice, list_specificity, list_precision, list_recall, list_accuracy, list_jaccard = [], [], [], [], [], []
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(
                device)
            test_pred = trainer.val_op(test_data, test_target)
            for pred, gt in zip(test_pred[0], lab):
                pred = torch.sigmoid(pred.cpu().squeeze())
                pred = (pred>0.5).cpu()
                # gt中非0的像素值全部置为1
                gt = (gt>127).cpu()
                metrics.update(pred, gt)
                dice, specificity, precision, recall, accuracy, jaccard = evaluate_one_img(pred.cpu().detach().numpy(), gt.cpu().detach().numpy())
                list_dice.append(dice)
                list_specificity.append(specificity)
                list_precision.append(precision)
                list_recall.append(recall)
                list_accuracy.append(accuracy)
                list_jaccard.append(jaccard)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            save_pred = torch.zeros((480, 640))
            # save_pred = torch.zeros((480 * 2, 640))
            save_pred[:480, :] = test_pred
            # save_pred[480:, :] = lab.cpu().squeeze()
            save_name = os.path.join(save_path, os.path.split(names[1])[1])
            save_pred = save_pred.numpy() * 255
            cv2.imwrite(save_name, save_pred.astype(np.uint8))
        results_metrics = metrics.compute_metrics()
        print(("Dice: %.4f, Specificity: %.4f, Precision: %.4f, Recall: %.4f, Accuracy: %.4f, Jaccard: %.4f" \
                  % (results_metrics["Dice"], results_metrics["Specificity"], results_metrics["Precision"], results_metrics["Recall"], results_metrics["Accuracy"], results_metrics["Jaccard"])))
        test_dice = sum(list_dice) / len(list_dice)
        test_specificity = sum(list_specificity) / len(list_specificity)
        test_precision = sum(list_precision) / len(list_precision)
        test_recall = sum(list_recall) / len(list_recall)
        test_accuracy = sum(list_accuracy) / len(list_accuracy)
        test_jaccard = sum(list_jaccard) / len(list_jaccard)
        print('Test Jaccard: {:.4f}'.format(test_jaccard))
        print('Test Dice: {:.4f}'.format(test_dice))
        print('Test Precision: {:.4f}'.format(test_precision))
        print('Test Recall: {:.4f}'.format(test_recall))
        print('Test Accuracy: {:.4f}'.format(test_accuracy))
        print('Test Specificity: {:.4f}'.format(test_specificity))

if __name__ == '__main__':
    test_data_path = 'path/to/your/Dataset/00_List/test.txt'
    save_path = './deepcrack_results/'
    pretrained_model = './checkpoints/DeepCrack.pth'
    test(test_data_path, save_path, pretrained_model)
