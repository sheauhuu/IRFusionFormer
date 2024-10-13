from pprint import pprint
import os
import setproctitle

class Config:
    name = 'DeepCrack'

    gpu_id = '0'

    setproctitle.setproctitle("%s" % name)

    # path
    train_data_path = 'path/to/your/Dataset/00_List/train_val.txt'
    val_data_path = 'path/to/your/Dataset/00_List/test.txt'
    checkpoint_path = 'checkpoints'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20

    # training
    epoch = 200
    pretrained_model = './checkpoints/DeepCrack_CT260_FT1.pth'
    weight_decay = 0.0001
    lr_decay = 0.1
    lr = 1e-4
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
    train_batch_size = 2
    val_batch_size = 1
    test_batch_size = 1

    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1

    # visdom
    vis_env = 'DeepCrack'
    port = 8097
    vis_train_loss_every = 90
    vis_train_acc_every = 90
    vis_train_img_every = 90
    val_every = (358 // train_batch_size + 1) * 5  # 358 is the number of train samples
    val_every_epoch = 5
    image_size = (480, 480)

    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
