from data.augmentation import augCompose, RandomBlur, RandomColorJitter
from data.dataset import readIndex, dataReadPip, loadedDataset
from tqdm import tqdm
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
from config import Config as cfg
import numpy as np
import torch
import os
import cv2
import sys
import logging
from tools.metrics import Metrics
from albumentations import Compose, Resize

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # ----------------------- dataset ----------------------- #

    data_augment_op = augCompose(transforms=[[RandomColorJitter, 0.5], [RandomBlur, 0.2]])

    resize = Compose([Resize(cfg.image_size[0], cfg.image_size[1], always_apply=True)])

    train_pipline = dataReadPip(transforms=data_augment_op, resize=resize)

    test_pipline = dataReadPip(transforms=None, resize=resize)

    train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=True), preprocess=train_pipline)

    test_dataset = loadedDataset(readIndex(cfg.val_data_path), preprocess=test_pipline)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)

    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.val_batch_size,
                                             shuffle=False, num_workers=4, drop_last=True)
    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)
    metrics = Metrics(2, 255, 'cpu')
    best_dice = 0
    best_dice_epoch = -1

    if cfg.pretrained_model:
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('load checkpoint: %s' % cfg.pretrained_model)
    try:

        for epoch in range(1, cfg.epoch):
            logging.info('Start Epoch %d ...' % epoch)
            model.train()

            # ---------------------  training ------------------- #
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            for idx, (img, lab) in bar:
                data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                pred = trainer.train_op(data, target)

            if epoch % cfg.val_every_epoch == 0:
                logging.info('Start Val %d ....' % idx)
                # -------------------- val ------------------- #
                model.eval()

                bar.set_description('Epoch %d --- Evaluation --- :' % epoch)
                with torch.no_grad():
                    for idx, (img, lab) in enumerate(val_loader, start=1):
                        val_data, val_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(
                            torch.cuda.FloatTensor).to(device)
                        val_pred = trainer.val_op(val_data, val_target)
                        pred = val_pred[0]
                        pred = torch.sigmoid(pred)
                        pred[pred > cfg.acc_sigmoid_th] = 1
                        pred[pred <= cfg.acc_sigmoid_th] = 0
                        target = val_target
                        metrics.update(pred.squeeze(0).long().cpu(), target.long().cpu())
                    results_metrics = metrics.compute_metrics()
                    logging.info(("Dice: %.4f, Specificity: %.4f, Precision: %.4f, Recall: %.4f, Accuracy: %.4f, Jaccard: %.4f" \
                            % (results_metrics["Dice"], results_metrics["Specificity"], results_metrics["Precision"], results_metrics["Recall"], results_metrics["Accuracy"], results_metrics["Jaccard"])))
                    if results_metrics["Dice"] > best_dice:
                        best_dice = results_metrics["Dice"]
                        best_dice_epoch = epoch
                        trainer.saver.save(model, tag='best_epoch(%d)_dice(%0.5f)' % (
                            epoch, best_dice))
                        logging.info('Save Model -best_epoch(%d)_dice(%0.5f)' % (
                            epoch, best_dice))
                model.train()

    except KeyboardInterrupt:

        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        logging.info('Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        logging.info('Training End!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    main()
