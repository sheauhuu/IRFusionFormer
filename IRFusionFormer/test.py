import os
import yaml
from IRFusionFormer.train import InfraredModel, arg_parse, set_random_seed, print_options
# from train_pl_aux import InfraredModel, arg_parse, set_random_seed, print_options
import lightning as L

from models import APFNet, DMFNet, crackformer, FusionFormer, FusionFormer_AS, UNet, NestedUNet, deeplabv3plus_resnet50
import torch

if __name__ == '__main__':
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    # Set GPU ID
    gpus = ','.join([str(i) for i in configs['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Fix seed (for repeatability)
    set_random_seed(seed=configs['seed'])

    # Open log file
    # open_log(args, configs)
    # logging.info(args)
    # monai.config.print_config()
    # print_options(configs)
    checkpoint_path = configs['test_ckpt_path']
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
    model = InfraredModel.load_from_checkpoint(checkpoint_path, model=net, configs=configs)
    # print(print(isinstance(model, L.LightningModule)))
    # model = InfraredModel(net, configs)
    # model.load_state_dict(torch.load(checkpoint_path))
    # 创建 Trainer 实例
    trainer = L.Trainer(
        devices=configs['GPUs'],  # 确保这里的配置与训练时一致
        logger=False,  # 测试时可能不需要 logger
    )
    trainer.test(model)