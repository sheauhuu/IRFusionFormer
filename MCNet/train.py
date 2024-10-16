from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader, get_val_loader
from network import MCNet
from furnace.datasets.ir_crack.IRT_Crack import IRT_Crack

from furnace.utils.init_func import init_weight, group_weight
from furnace.utils.pyt_utils import all_reduce_tensor
from furnace.engine.lr_policy import PolyLR
from furnace.engine.logger import get_logger
from furnace.engine.engine import Engine
from furnace.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, EdgeLoss
from furnace.utils.metrics import Metrics

# try:
#     from apex.parallel import SyncBatchNorm, DistributedDataParallel
# except ImportError:
#     raise ImportError(
#         "Please install apex from https://www.github.com/nvidia/apex .")

logger = get_logger()

torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

parser = argparse.ArgumentParser()

metrics = Metrics(config.num_classes, 255, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

with Engine(custom_parser=parser) as engine:

    engine.distributed = False

    args = parser.parse_args()

    cudnn.benchmark = True
    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, IRT_Crack)
    val_loader, val_sampler = get_val_loader(engine, IRT_Crack)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)  # 默认使用交叉熵损失
    edge_criterion = EdgeLoss(ignore_label=255)
    '''
    min_kept = int(config.batch_size // len(
        engine.devices) * config.image_height * config.image_width // 16)
    criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                       min_kept=min_kept,
                                       use_weight=False)
    '''

    # if engine.distributed:
    #     logger.info('Use the Multi-Process-SyncBatchNorm')
    #     BatchNorm2d = SyncBatchNorm
    # else:
    #     BatchNorm2d = nn.BatchNorm2d
    BatchNorm2d = nn.BatchNorm2d

    model = MCNet(config.num_classes, criterion=criterion,
                  edge_criterion=edge_criterion, pretrained_model=config.pretrained_model,
                  norm_layer=BatchNorm2d)
    init_weight(model.business_layer[0:-1], nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr * 10, smooth_dilation=True)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # if engine.distributed:
    #     model = DistributedDataParallel(model)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()

    for epoch in range(engine.state.epoch, config.nepochs):
        print('Epoch:', epoch)
        totalloss = 0
        edge_total_loss = 0
        totalonelevel = 0
        totaltwolevel = 0
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            engine.update_iteration(epoch, idx)

            # minibatch = dataloader.next()
            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            edge_gts = minibatch['aux_label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            edge_gts = edge_gts.cuda(non_blocking=True)
            loss, edge_loss, onelevel_segloss, twolevel_segloss = model(imgs, gts, edge_gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss,
                                                world_size=engine.world_size)
                reduce_edge_loss = all_reduce_tensor(edge_loss,
                                                     world_size=engine.world_size)
                onelevel_segloss = all_reduce_tensor(onelevel_segloss,
                                                    world_size=engine.world_size)
                twolevel_segloss = all_reduce_tensor(twolevel_segloss,
                                                    world_size=engine.world_size)
            else:
                reduce_loss = loss
                reduce_edge_loss = edge_loss
                onelevel_segloss = onelevel_segloss
                twolevel_segloss = twolevel_segloss

            optimizer.zero_grad()
            # Floating point exception (core dumped)
            assert torch.isfinite(loss).all(), "Loss contains NaN or Inf."
            print('loss:', loss.float())
            loss.backward()
            optimizer.step()

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

            totalloss = totalloss + reduce_loss.item()
            edge_total_loss = edge_total_loss + reduce_edge_loss.item()
            totalonelevel = totalonelevel + onelevel_segloss.item()
            totaltwolevel = totaltwolevel + twolevel_segloss.item()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item() \
                        + ' totalloss=%.2f' % totalloss \
                        + ' edgeloss=%.2f' % edge_total_loss \
                        + ' level_1_loss=%.2f' % totalonelevel \
                        + ' level_2_loss =%.2f ' % totaltwolevel

            pbar.set_description(print_str, refresh=False)

        # if (epoch >= config.nepochs - 20) or (
        #         epoch % config.snapshot_iter == 0):
        #     if engine.distributed and (engine.local_rank == 0):
        #         engine.save_and_link_checkpoint(config.snapshot_dir,
        #                                         config.log_dir,
        #                                         config.log_dir_link)
        #     elif not engine.distributed:
        #         engine.save_and_link_checkpoint(config.snapshot_dir,
        #                                         config.log_dir,
        #                                         config.log_dir_link)
        if epoch % config.val_epoch == 0:
            model.eval()
            with torch.no_grad():
                for idx, minibatch in enumerate(val_loader):
                    imgs = minibatch['data']
                    gts = minibatch['label']
                    edge_gts = minibatch['aux_label']

                    imgs = imgs.cuda(non_blocking=True)
                    gts = gts.cuda(non_blocking=True)
                    # edge_gts = edge_gts.cuda(non_blocking=True)
                    pred = model(imgs)
                    

                    metrics.update(model(imgs), gts)
            results_metrics = metrics.compute_metrics()
            print("Dice: %.4f, Specificity: %.4f, Precision: %.4f, Recall: %.4f, Accuracy: %.4f, Jaccard: %.4f" \
                  % (results_metrics["Dice"], results_metrics["Specificity"], results_metrics["Precision"], results_metrics["Recall"], results_metrics["Accuracy"], results_metrics["Jaccard"]))
            metrics.reset()
            model.train()