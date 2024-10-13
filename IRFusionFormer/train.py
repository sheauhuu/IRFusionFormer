import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from data.ircrack_dataset import get_crack_dataset
from util.lr_scheduler import LinearWarmupCosineAnnealingLR
from util.tools import make_necessary_dirs, print_options
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util import misc2
from models import APFNet, DMFNet, crackformer, FusionFormer, FusionFormer_AS, UNet, NestedUNet, deeplabv3plus_resnet50
from util.soft_skeleton import SoftSkeletonize
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from util.metrics import Metrics
import logging
import cv2
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def save_image(tensor_pred, tensor_gt, tensor_rgb, tensor_infrared, filename):
    tensor_pred = tensor_pred.mul(255).byte()
    tensor_gt = tensor_gt.mul(255).byte()
    pred = tensor_pred.squeeze(0).cpu().numpy()
    gt = tensor_gt.squeeze(0).cpu().numpy()
    rgb = tensor_rgb.permute(1, 2, 0).cpu().numpy()
    rgb = denormalize_image(rgb)
    infrared = tensor_infrared.permute(1, 2, 0).cpu().numpy()
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(rgb)
    plt.title('rgb')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(infrared, cmap='gray')
    plt.title('infrared')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(pred, cmap='gray')
    plt.title('pred')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(gt, cmap='gray')
    plt.title('gt')
    plt.axis('off')
    plt.savefig(filename)

def one_hot(label, num_classes=2):
    one_hot = torch.zeros(label.size(0), num_classes, label.size(2), label.size(3)).to(label.device)
    # clamp to 2 channels
    # print(label.shape)
    # print((label==1).float().shape)
    # print(one_hot[:, 0, ...].shape)
    if label.size(0) != 1:
        one_hot[:, 0, ...] = (label == 0).squeeze(1).float()
        one_hot[:, 1, ...] = (label == 1).squeeze(1).float()
    else:
        one_hot[:, 0, ...] = (label == 0).float()
        one_hot[:, 1, ...] = (label == 1).float()
    # print(torch.sum(torch.abs(one_hot[:, 0, ...] - one_hot[:, 1, ...])))
    # print(torch.sum(torch.ones_like(one_hot)))
    # print(one_hot.shape)
    return one_hot

class soft_cldice(nn.Module):
    def __init__(self, iter_=10, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        # y_pred = F.softmax(y_pred, dim=1)
        y_pred = F.sigmoid(y_pred)
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True,):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, inputs, targets):
        targets = targets.long()
        loss = F.cross_entropy(inputs, targets, reduction='mean')
        # print('inputs shape', inputs.shape)
        # print('targets shape', targets.shape)
        # inputs.shape == [bs, 2, h, w]
        # targets.shape == [bs, 1, h, w]
        return loss

class BCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True,):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        if inputs.shape[1] == 2:
            inputs = inputs[:, 1, :, :]
        elif inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        # inputs = inputs[:, 1, :, :]
        targets = targets.float()
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        return BCE

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        inputs = F.softmax(inputs, dim=1)  
        if inputs.shape[1] == 2:
            inputs = inputs[:, 1, :, :]
        elif inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        # inputs = inputs[:, 1, :, :]
        # inputs = inputs.reshape(-1)
        # targets = targets.reshape(-1)
        #first compute binary cross-entropy
        targets = targets.float() 
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        num = targets.size(0)
        # inputs = F.softmax(inputs, dim=1)
        # inputs = torch.argmax(inputs, dim=1).float()
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        if inputs.shape[1] == 2:
            inputs = inputs[:,1,:,:]
        elif inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        # inputs = inputs[:,1,:,:]
        inputs = inputs.view(num, -1)
        targets = targets.view(num, -1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        
        return 1 - dice
     # 改写这个 加入weighted
class combine_loss(nn.Module):
    def __init__(self, dice_weight=0.1, bce_weight=0.1, focal_weight=0.1, ce_weight=0.1, soft_cldice_weight=0.1):
        super(combine_loss, self).__init__()
        self.soft_cldice_weight = soft_cldice_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = BCELoss()
        self.focal_loss = FocalLoss()
        self.ce_loss = CrossEntropyLoss()
        self.soft_cldice_loss = soft_cldice(iter_=10, smooth=1., exclude_background=True)
    
    def forward(self, inputs, targets):
        assert len(inputs.shape) == len(targets.shape) == 4
        dice = self.dice_loss(inputs, targets)
        # dice = self.dice_loss(inputs, one_hot(targets))
        bce = self.bce_loss(inputs, targets.squeeze(1))
        focal = self.focal_loss(inputs, targets.squeeze(1))
        if self.ce_weight != 0:
            ce = self.ce_loss(inputs, targets.squeeze(1))
        else:
            ce = torch.tensor(0)
        # soft_cldice = self.soft_cldice_loss(y_pred=inputs, y_true=one_hot(targets))
        if self.soft_cldice_weight != 0:
            soft_cldice = self.soft_cldice_loss(y_pred=inputs, y_true=one_hot(targets))
        else:
            soft_cldice = torch.tensor(0)

        # print('dice loss', dice, 'bce loss', bce, 'focal loss', focal, 'ce loss', ce, 'soft_cldice loss', soft_cldice)
        # print('dice loss', dice)
        # print('bce loss', bce)
        # print('focal loss', focal)
        # print('ce loss', ce)
        # print('soft_cldice loss', soft_cldice)
        loss = self.dice_weight * dice + self.bce_weight * bce + self.focal_weight * focal + self.ce_weight * ce + self.soft_cldice_weight * soft_cldice
        return loss

def arg_parse() -> argparse.ArgumentParser.parse_args :
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', default='./configs/train_valid4_universeg_no_label.yaml',
                        type=str, help='load the config file')
    args = parser.parse_args()
    return args

def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class InfraredModel(L.LightningModule):
    def __init__(self, model, configs):
        super(InfraredModel, self).__init__()
        self.model = model.to('cuda')
        self.configs = configs
        self.epochs = configs['epochs']
        self.save_path = configs['ckpt_path']
        self.lr = configs['lr']
        self.weight_decay = configs['weight_decay']
        self.num_workers = configs['num_workers']
        self.valid_epoch_freq = configs['valid_epoch_freq']
        self.save_epoch_freq = configs['save_epoch_freq']
        self.save_iter_freq = configs['save_iter_freq']
        self.print_iter_freq = configs['print_iter_freq']
        self.input_type = configs['input_type']
        self.aux_loss_weight = configs['aux_loss_weight']
        # self.loss_function, self.loss_weight = self.loss_function_config(configs)
        self.best_dice_valid = 0.0
        self.best_dice_valid_epoch = 0.0
        n_classes = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = Metrics(n_classes, 255, device)
        self.loss = combine_loss(dice_weight=configs['loss_type']['Dice'], bce_weight=configs['loss_type']['BCE'], 
                                 focal_weight=configs['loss_type']['Focal'], ce_weight=configs['loss_type']['CE'], 
                                 soft_cldice_weight=configs['loss_type']['SoftCLDice'])
        self.celoss = CrossEntropyLoss()

    def forward(self, infrareds, rgbs):
        with torch.cuda.amp.autocast():
            outputs = self.model(infrareds, rgbs)
        return outputs

    def configure_optimizers(self):
        if self.configs['model'] == 'APFNet' or 'DMFNet':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=self.epochs)
        elif self.configs['model'] == 'UNet':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.99)
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=self.epochs)
        elif self.configs['model'] == 'NestedUNet':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=self.epochs)
        elif self.configs['model'] == 'Crackformer':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100], gamma=0.1)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=self.weight_decay)
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=self.epochs)

        return [optimizer], [scheduler]
    
    def evaluate_one_img(self, pred, gt):
        dice = misc2.dice(pred, gt)
        specificity = misc2.specificity(pred, gt)
        jaccard = misc2.jaccard(pred, gt)
        precision = misc2.precision(pred, gt)
        recall = misc2.recall(pred, gt)
        f_measure = misc2.fscore(pred, gt)

        return dice, specificity, precision, recall, f_measure, jaccard

    # def loss_function_config(self, configs):
    #     loss_function = []
    #     loss_weight = []
    #     if 'Dice' in configs['loss_type'].keys():
    #         dice_weight = configs['loss_type']['Dice']
    #         dice_loss = DiceLoss()
    #         loss_function.append(dice_loss)
    #         loss_weight.append(dice_weight)
    #     if 'BCE' in configs['loss_type'].keys():
    #         bce_weight = configs['loss_type']['BCE']
    #         bce_loss = BCELoss()
    #         loss_function.append(bce_loss)
    #         loss_weight.append(bce_weight)
    #     if 'Focal' in configs['loss_type'].keys():
    #         focal_weight = configs['loss_type']['Focal']
    #         focal_loss = FocalLoss()
    #         loss_function.append(focal_loss)
    #         loss_weight.append(focal_weight)
    #     if 'CE' in configs['loss_type'].keys():
    #         ce_weight = configs['loss_type']['CE']
    #         ce_loss = CrossEntropyLoss()
    #         loss_function.append(ce_loss)
    #         loss_weight.append(ce_weight)
        
    #     return loss_function, loss_weight

    # def compute_loss(self, outputs, labels):
    #     for j in range(len(self.loss_function)):
    #         if j == 0:
    #             loss = self.loss_weight[j] * self.loss_function[j](outputs, labels.float())
    #         else:
    #             loss += self.loss_weight[j] * self.loss_function[j](outputs, labels.float())

    #     return loss

    def training_step(self, batch, batch_idx):
        self.model.train()
        torch.set_grad_enabled(True)
        labels = batch['gt']
        if self.input_type == 'rgb' and self.configs['model'] != 'Crackformer':
            rgbs = batch['rgb']
            outputs = self.model(rgbs)
            loss = self.loss(outputs, labels)
        elif self.input_type == 'rgb' and self.configs['model'] == 'Crackformer':
            rgbs = batch['rgb']
            outputs = self.model(rgbs)
            outputs = outputs[-1]
            # outputs = torch.sigmoid(outputs)
            # loss = self.model.calculate_loss(outputs, labels)
            loss = self.loss(outputs, labels)
        elif self.input_type == 'infrared':
            infrareds = batch['infrared']
            outputs = self.model(infrareds)
            loss = self.loss(outputs, labels)
        elif self.input_type == 'fusion' and self.configs['model'] != 'FusionFormer' and self.configs['model'] != 'FusionFormer_AS':
            rgbs, infrareds = batch['rgb'], batch['infrared']
            outputs = self.model(infrareds, rgbs)
            loss = self.loss(outputs, labels)
        elif self.input_type == 'fusion' and (self.configs['model'] == 'FusionFormer' or self.configs['model'] == 'FusionFormer_AS'):
            rgbs, infrareds = batch['rgb'], batch['infrared']
            outputs = self.model(infrared_images=infrareds, rgb_images=rgbs)
            output = outputs[0]
            aux_outputs = outputs[1]
            loss = self.loss(output, labels)
            aux_loss = self.loss(aux_outputs, labels)
            self.log('aux_loss', aux_loss)
            loss += self.aux_loss_weight * aux_loss
        # if self.configs['model'] == 'Crackformer':
        #     outputs = outputs[-1]
        #     loss = self.loss(outputs, labels)
        # loss = self.compute_loss(outputs, labels.squeeze(1))
        # print('outputs:', outputs.shape)
        # print('labels:', labels.squeeze(1).shape)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)
        rgbs, infrareds, labels, names = batch['rgb'], batch['infrared'], batch['gt'], batch['name']
        if self.input_type == 'fusion':
            outputs = self.model(infrared_images =infrareds, rgb_images = rgbs)
            val_loss = self.loss(outputs, labels)
            preds = outputs.softmax(dim=1)
            preds = preds.argmax(dim=1)
        elif self.input_type == 'rgb' and self.configs['model'] != 'Crackformer' and self.configs['model'] != 'DMFNet':
            outputs = self.model(rgbs)
            val_loss = self.loss(outputs, labels)
            preds = outputs.softmax(dim=1)
            preds = preds.argmax(dim=1)
        elif self.input_type == 'rgb' and self.configs['model'] == 'Crackformer':
            outputs = self.model(rgbs)
            outputs = outputs[-1]
            # preds = torch.sigmoid(outputs)
            # val_loss = self.model.calculate_loss(outputs, labels)
            val_loss = self.loss(outputs, labels)
            # print('preds shape:', preds.shape)
            # preds = preds.squeeze(0)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).squeeze(1)
        elif self.input_type == 'rgb' and self.configs['model'] == 'DMFNet':
            outputs = self.model(rgbs)
            # print('outputs shape:', outputs.shape)
            # print('labels shape:', labels.shape)
            val_loss = self.celoss(outputs, labels.squeeze(0).long())
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).squeeze(1)
            
        self.log('val_loss', val_loss)
        labels = labels.squeeze(1)
        self.metrics.update(preds, labels)
        # svae the valid img results
        if not os.path.exists(os.path.join(self.save_path, 'valid_{}'.format(self.current_epoch))):
            os.makedirs(os.path.join(self.save_path, 'valid_{}'.format(self.current_epoch)))
        for pred, gt, rgb, infrared, name in zip(preds, labels, rgbs, infrareds, names):
            filename = 'sample_{}.png'.format(name)
            save_image(pred, gt, rgb, infrared, os.path.join(self.save_path, 'valid_{}'.format(self.current_epoch), filename))

    def on_validation_epoch_end(self):
        results_metrics = self.metrics.compute_metrics()
        logging.info("Dice: %.4f, Specificity: %.4f, Precision: %.4f, Recall: %.4f, Accuracy: %.4f, Jaccard: %.4f" \
                  % (results_metrics["Dice"], results_metrics["Specificity"], results_metrics["Precision"], results_metrics["Recall"], results_metrics["Accuracy"], results_metrics["Jaccard"]))
        self.log('val_dice', results_metrics["Dice"])
        self.log('val_specificity', results_metrics["Specificity"])
        self.log('val_precision', results_metrics["Precision"])
        self.log('val_recall', results_metrics["Recall"])
        self.log('val_accuracy', results_metrics["Accuracy"])
        self.log('val_jaccard', results_metrics["Jaccard"])
        if results_metrics["Dice"] > self.best_dice_valid:
            self.best_dice_valid = results_metrics["Dice"]
            self.best_dice_valid_epoch = self.current_epoch
            logging.info('Best dice valid: %.4f' % (self.best_dice_valid))
            logging.info('Best dice valid epoch: %d' % self.best_dice_valid_epoch)
        self.log('best_dice_valid:', self.best_dice_valid)
        self.metrics.reset()
    
    def test_step(self, batch, batch_idx):
        # print('Evaluating...')
        # self.model.eval()
        # rgbs, infrareds, labels = batch['rgb'], batch['infrared'], batch['gt']
        # outputs = self.model(infrared_images =infrareds, rgb_images = rgbs)
        # self.metrics.update(outputs.softmax(dim=1), labels)
        # self.model.eval()
        # torch.set_grad_enabled(False)
        # rgbs, infrareds, labels, names = batch['rgb'], batch['infrared'], batch['gt'], batch['name']
        # if self.input_type == 'fusion':
        #     outputs = self.model(infrared_images =infrareds, rgb_images = rgbs)
        #     val_loss = self.loss(outputs, labels)
        #     preds = outputs.softmax(dim=1)
        #     preds = preds.argmax(dim=1)
        # elif self.input_type == 'rgb' and self.configs['model'] != 'Crackformer':
        #     outputs = self.model(rgbs)
        #     val_loss = self.loss(outputs, labels)
        #     preds = outputs.softmax(dim=1)
        #     preds = preds.argmax(dim=1)
        # elif self.input_type == 'rgb' and self.configs['model'] == 'Crackformer':
        #     outputs = self.model(rgbs)
        #     outputs = outputs[-1]
        #     preds = torch.sigmoid(outputs)
        #     val_loss = self.model.calculate_loss(outputs, labels)
        #     # print('preds shape:', preds.shape)
        #     # preds = preds.squeeze(0)
        #     preds = (preds > 0.5).squeeze(1)
        # labels = labels.squeeze(1)
        # self.metrics.update(preds, labels)
        self.model.eval()
        torch.set_grad_enabled(False)
        rgbs, infrareds, labels, names = batch['rgb'], batch['infrared'], batch['gt'], batch['name']
        if self.input_type == 'fusion':
            outputs = self.model(infrared_images =infrareds, rgb_images = rgbs)
            # val_loss = self.loss(outputs, labels)
            preds = outputs.softmax(dim=1)
            preds = preds.argmax(dim=1)
        elif self.input_type == 'rgb' and self.configs['model'] != 'Crackformer' and self.configs['model'] != 'DMFNet':
            outputs = self.model(rgbs)
            # val_loss = self.loss(outputs, labels)
            preds = outputs.softmax(dim=1)
            preds = preds.argmax(dim=1)
        elif self.input_type == 'rgb' and self.configs['model'] == 'Crackformer':
            outputs = self.model(rgbs)
            outputs = outputs[-1]
            # preds = torch.sigmoid(outputs)
            # val_loss = self.model.calculate_loss(outputs, labels)
            # val_loss = self.loss(outputs, labels)
            # print('preds shape:', preds.shape)
            # preds = preds.squeeze(0)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).squeeze(1)
        elif self.input_type == 'rgb' and self.configs['model'] == 'DMFNet':
            outputs = self.model(rgbs)
            # print('outputs shape:', outputs.shape)
            # print('labels shape:', labels.shape)
            # val_loss = self.celoss(outputs, labels.squeeze(0).long())
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).squeeze(1)
            
        # self.log('val_loss', val_loss)
        labels = labels.squeeze(1)
        self.metrics.update(preds, labels)
        # svae the valid img results
        if not os.path.exists(os.path.join(self.save_path, self.configs['model'],'Test')):
            os.makedirs(os.path.join(self.save_path, self.configs['model'], 'Test'))
        for pred, gt, rgb, infrared, name in zip(preds, labels, rgbs, infrareds, names):
            filename = 'sample_{}.png'.format(name)
            img_pred = pred.mul(255).byte()
            img_pred = img_pred.squeeze(0).cpu().numpy()
            cv2.imwrite(os.path.join(self.save_path, self.configs['model'], 'Test', filename), img_pred)
    
    def on_test_epoch_end(self):
        results_metrics = self.metrics.compute_metrics()
        # print("Dice: %.4f, Specificity: %.4f, Precision: %.4f, Recall: %.4f, Accuracy: %.4f, Jaccard: %.4f" \
        #           % (results_metrics["Dice"], results_metrics["Specificity"], results_metrics["Precision"], results_metrics["Recall"], results_metrics["Accuracy"], results_metrics["Jaccard"]))
        print("Dice: %.4f, Jaccard: %.4f, Accuracy: %.4f, Precision: %.4f, Specificity: %.4f, Recall: %.4f" \
                    % (results_metrics["Dice"], results_metrics["Jaccard"], results_metrics["Accuracy"], results_metrics["Precision"], results_metrics["Specificity"], results_metrics["Recall"]))
        # open text file
        with open('./test-final.log', 'a') as f:
            # add text to file
            f.write(self.configs['name'] + '\n')
            f.write("Dice: %.4f, Jaccard: %.4f, Accuracy: %.4f, Precision: %.4f, Specificity: %.4f, Recall: %.4f\n" \
                    % (results_metrics["Dice"], results_metrics["Jaccard"], results_metrics["Accuracy"], results_metrics["Precision"], results_metrics["Specificity"], results_metrics["Recall"]))
        self.metrics.reset()
    
    def train_dataloader(self):
        train_loader =  get_crack_dataset(self.configs, 'train')
        return train_loader
    
    def val_dataloader(self):
        val_loader = get_crack_dataset(self.configs, 'val')
        return val_loader
    
    def test_dataloader(self):
        test_loader = get_crack_dataset(self.configs, 'val')
        return test_loader

if __name__ == "__main__":


    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    configs['ckpt_path'] = os.path.join(configs['ckpt_path'], configs['model'], "{}-{}".format(time.strftime('%Y%m%d%H%M', time.localtime(time.time())), configs['name']))

    # Set GPU ID
    gpus = ','.join([str(i) for i in configs['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Fix seed (for repeatability)
    set_random_seed(seed=configs['seed'])

    # Output folder and save fig folder
    configs = make_necessary_dirs(configs)

    file_name = print_options(configs)

    model_name = configs['model']
    assert model_name in ['APFNet', 'DMFNet', 'Crackformer', 'FusionFormer', 'FusionFormer_AS', 'UNet', 'NestedUNet', 'DeepLabV3Plus']
    if model_name == 'APFNet':
        net = APFNet(3,1)
    elif model_name == 'DMFNet':
        net = DMFNet(3,1)
    elif model_name == 'Crackformer':
        net = crackformer()
    elif model_name == 'UNet':
        net = UNet(num_classes=2)
    elif model_name == 'NestedUNet':
        net = NestedUNet(num_classes=2)
    elif model_name == 'DeepLabV3Plus':
        net = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
    elif model_name == 'FusionFormer' or model_name == 'FusionFormer_AS':
        if 'infrared_encoder' in configs.keys():
            assert configs['infrared_encoder'] in ['Resnet18', 'Resnet34', 'Resnet50']
            infrared_encoder = configs['infrared_encoder']
        else:
            infrared_encoder = 'Resnet18'
        if 'fusion_stage' in configs.keys():
            assert len(configs['fusion_stage']) <= 3
            assert all([i in [1, 2, 3] for i in configs['fusion_stage']])
            fusion_stage = configs['fusion_stage']
        else:
            fusion_stage = [1, 2, 3]
        if model_name == 'FusionFormer':
            net = FusionFormer(num_classes=2, infrared_channels=1, image_size=configs['img_size'][0], heads=configs['heads'], infrared_encoder=infrared_encoder)
        elif model_name == 'FusionFormer_AS':
            net = FusionFormer_AS(num_classes=2, infrared_channels=1, image_size=configs['img_size'][0], heads=configs['heads'], infrared_encoder=infrared_encoder, fusion_stage=fusion_stage)
    # from torchinfo import summary
    # net = net.to('cuda')
    # summary(net, input_size=(1, 3, 480, 480))
    model = InfraredModel(net, configs)

    wandb_logger = WandbLogger(project="infrared", log_model=True, name='{}_{}_{}'.format(model_name, 'Loss', time.strftime('%Y%m%d%H%M', time.localtime(time.time()))))

    # 配置日志记录
    logging.basicConfig(filename=file_name, level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        mode = "max",
        dirpath= configs['ckpt_path'],
        filename=model_name + '-epoch{epoch:02d}-Dice-{val_dice:.4f}',
        auto_insert_metric_name=False,
        every_n_epochs=configs['valid_epoch_freq'],
        save_top_k=1,
        save_last=True
        )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='max'
        )
    if 'early_stop' in configs.keys() and configs['early_stop']:
        callbacks = [checkpoint_callback, lr_monitor_callback, early_stop_callback]
    else:
        callbacks = [checkpoint_callback, lr_monitor_callback]

    trainer = L.Trainer(
        check_val_every_n_epoch = configs['valid_epoch_freq'],
        max_epochs  = configs['epochs'],
        devices     = configs['GPUs'],
        # logger      = wandb_logger,
        logger= False,
        # callbacks   = callbacks,
        # callbacks=[checkpoint_callback, early_stop_callback],
        callbacks=[checkpoint_callback],
        # num_sanity_val_steps=0,
        # precision='bf16',
    )
    trainer.fit(model)