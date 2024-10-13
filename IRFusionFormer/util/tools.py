import torch
import os
import random
import numpy as np
import logging
import torchvision
import matplotlib
import matplotlib.pyplot as plt


def plot_progress(train_loss_dict, valid_loss_dict, valid_dice_dict, configs):
    font = {'weight': 'normal', 'size': 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(30, 24))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    x_train_values, train_loss = list(train_loss_dict.keys()), list(train_loss_dict.values())
    ax.plot(x_train_values, train_loss, color='b', ls='-', label="loss_tr")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    if valid_loss_dict != {} and valid_dice_dict != {}:
        x_valid_values, valid_loss = list(valid_loss_dict.keys()), list(valid_loss_dict.values())
        ax.plot(x_valid_values, valid_loss, color='r', ls='-', label="loss_val, train=False")

        x_valid_values, valid_dice = list(valid_dice_dict.keys()), list(valid_dice_dict.values())
        ax2.plot(x_valid_values, valid_dice, color='g', ls='--', label="evaluation metric")

        ax2.set_ylabel("evaluation metric")
        ax2.legend(loc=9)

    ax.legend()

    fig.savefig(os.path.join(configs['ckpt_path'], "progress.png"))
    plt.close()


def save_images(visuals, configs, epoch, i):
    comp        = visuals['comp']
    output      = visuals['harmonized']
    real        = visuals['real']
    lesion_mask = visuals['mask']

    k = int(64 // 2)
    batch_comp_images_view1 = comp[..., k].cpu()
    batch_harm_images_view1 = output[..., k].cpu()
    batch_real_images_view1 = real[..., k].cpu()
    batch_mask_images_view1 = lesion_mask[..., k].cpu()

    batch_comp_images_view2 = comp[..., k, :].cpu()
    batch_harm_images_view2 = output[..., k, :].cpu()
    batch_real_images_view2 = real[..., k, :].cpu()
    batch_mask_images_view2 = lesion_mask[..., k, :].cpu()

    batch_comp_images_view3 = comp[..., k, :, :].cpu()
    batch_harm_images_view3 = output[..., k, :, :].cpu()
    batch_real_images_view3 = real[..., k, :, :].cpu()
    batch_mask_images_view3 = lesion_mask[..., k, :, :].cpu()

    slices = torch.cat((
        batch_comp_images_view1,
        batch_harm_images_view1,
        batch_real_images_view1,
        batch_mask_images_view1,
        batch_comp_images_view2,
        batch_harm_images_view2,
        batch_real_images_view2,
        batch_mask_images_view2,
        batch_comp_images_view3,
        batch_harm_images_view3,
        batch_real_images_view3,
        batch_mask_images_view3,
    ))
    image_path = os.path.join(configs['save_image_folder'], 'epoch{}_batch{}.png'.format(epoch, i))
    torchvision.utils.save_image(
        slices,
        image_path,
        nrow=configs['batch_size'],
        # normalize=True,
        # scale_each=True,
    )


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_options(configs):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in configs.items():
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    logging.info(message)

    # save to the disk
    file_name = os.path.join(configs['log_path'], '{}_configs.txt'.format(configs['phase']))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
    return file_name


def make_necessary_dirs(configs):
    # make ckpt dir if not exist
    mkdir(configs['ckpt_path'])

    # make fig dir if not exist
    save_image_folder = os.path.join(configs['ckpt_path'], 'fig')
    configs['save_image_folder'] = save_image_folder
    mkdir(save_image_folder)

    # make log dir if not exist
    log_path = os.path.join(configs['ckpt_path'], 'log')
    configs['log_path'] = log_path
    mkdir(log_path)

    # make validation results dir if not exist
    valid_results_folder = os.path.join(configs['ckpt_path'], 'valid_results')
    configs['valid_results_folder'] = valid_results_folder
    mkdir(valid_results_folder)

    return configs


def open_log(args, configs):
    # open the log file
    log_savepath = configs['log_path']
    log_name = args.config.split('/')[-1].split('.')[0]+'-yaml'
    if os.path.isfile(os.path.join(log_savepath, '{}.txt'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.txt'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.txt'.format(log_name)))


def initLogging(logFilename):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s-%(levelname)s] %(message)s',
        datefmt='%y-%m-%d %H:%M:%S',
        filename=logFilename,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def seed_reproducer(seed=2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def load_checkpoint(model: torch.nn.Module, path: str):
    if os.path.isfile(path):
        logging.info("=> loading checkpoint '{}'".format(path))
        
        # remap everthing onto CPU 
        state = torch.load(path, map_location=lambda storage, location: storage)

        # load weights
        model.load_state_dict(state['model'])
        logging.info("Loaded")
    else:
        model = None
        logging.info("=> no checkpoint found at '{}'".format(path))
    return model


def save_checkpoint(model: torch.nn.Module, optimizer, file_name: str, epoch: int):
    if optimizer:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        },file_name)
    else:
        torch.save({
            'model': model.state_dict(),
            'optimizer': None,
            'epoch': epoch,
        },file_name)
    # logging.info("save model to {}".format(file_name))


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def adjust_learning_rate(optimizer, current_epoch, configs):
    lr = poly_lr(current_epoch, int(configs['epochs']), float(configs['lr']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # logging.info('Change Learning Rate to {}'.format(lr))
    return lr


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor