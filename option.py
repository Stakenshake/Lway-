import argparse

parser = argparse.ArgumentParser()

''' 1. Gloab Settings '''
parser.add_argument('--wandb', action='store_true', default=False, help='Whether to use Wandb')
parser.add_argument('--wandb_project', type=str, default='Default', help='Wandb project name')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Whether to use cuda')
parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
parser.add_argument('--seed', type=int, default=123, help='seed')

''' 2. Saver '''
parser.add_argument('--record', type=str, default=None, help='record some settings in config')
parser.add_argument('--exp_name', type=str, default=None, help='save directory name')
parser.add_argument('--service', type=str, default='sr3', help='which service will use?')

''' 3. Data '''

parser.add_argument('--dataset', type=str, default='DF2K')
parser.add_argument('--train_batchsize', type=int, default=4, help='train batchsize')
parser.add_argument('--val_batchsize', type=int, default=4, help='Val batchsize')
parser.add_argument('--patch_size', type=int, nargs='+', default=[128, 128], help='Image Crop size (w,h) list of image crop sizes, with each item storing the crop size (should be a tuple).')
parser.add_argument('--rootpath', type=str, default='D:/Code/ComputerVision/LWay/Data/Datasets', help='Datset rootpath')
parser.add_argument('--scale_factor', type=int, default=2, help='scale factor.')
parser.add_argument('--sharp_factor', type=float, default=0.5, help='sharp factor.')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--repeat', type=int, default=1, help='dataset enlarge ratio')
parser.add_argument('--pin_memory', action='store_true', default=False, help='Whether to use pin_memory')


''' 4. Loss '''
parser.add_argument('--loss_type1', default='loss1', choices=['loss1','loss2','charb'], help='Loss function (ce or miou)')
parser.add_argument('--loss_type2', default='loss2', choices=['loss1','loss2','charb'], help='Loss function (ce or miou)')


''' 5. Learning rate '''
parser.add_argument('--lr', type=float, default=0.045, help='initial learning rate')
parser.add_argument('--lr_scheduler', default='step', choices=['cos','poly','step', 'fixed', 'clr', 'linear', 'hybrid'], help='Learning rate scheduler (fixed, clr, poly)')
parser.add_argument('--step_size', type=int, default=100, help='sharp factor.')
parser.add_argument('--step_gamma', type=float, default=0.5, help='sharp factor.')
parser.add_argument('--warmup_epochs', type=int, default=0, help='Warm Up epoch')
parser.add_argument('--ema', action='store_true', default=False, help='Whether to use Wandb')

''' 6. Optimizer '''
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer choose')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='SGD weight_decay')

''' 7. Train Settings '''
parser.add_argument('--start_epoch1', type=int, default=0, help='start epoch')
parser.add_argument('--epochs1', type=int, default=200, help='Total epochs')
parser.add_argument('--start_epoch2', type=int, default=0, help='start epoch')
parser.add_argument('--epochs2', type=int, default=200, help='Total epochs')
parser.add_argument('--epochs_encoder', type=int, default=100, help='number of epochs to train encoder.')
parser.add_argument('--delete_module', action='store_true', default=False,
                    help='Finetune the segmentation model')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Finetune the segmentation model')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='Finetune the segmentation model')

''' 8. Model '''
parser.add_argument('--model1', default='LRRec', help='Which model? basic= basic CNN model, res=resnet style)')#choices=['frame','edgevits','van']
parser.add_argument('--model2', default='SwinIR', help='Which model? basic= basic CNN model, res=resnet style)')#choices=['frame','edgevits','van']
parser.add_argument('--encoder_dim', type=int, default=512, help='the dimensionality of encoder.')
parser.add_argument('--scale', type=int, default=4)



''' 9. Msic '''
parser.add_argument('--vgg_weight', type=str, default=None, help='vgg_weight_path')

parser.add_argument('--epoch_eval', action='store_true', default=False, help='epoch eval mode')
parser.add_argument('--iter_eval', action='store_true', default=False, help='iter eval mode')
parser.add_argument('--eval_freq', type=int, default=200, help='iter eval frequency')
'''10.'''
parser.add_argument('--dataroot_lq', type=str, default='data/datasets/realsr', help='LQ 数据集路径')
parser.add_argument('--meta_info_file', type=str, default='', help='meta 信息文件路径')
parser.add_argument('--io_backend_type', type=str, default='', help='IO 后端类型')
parser.add_argument('--mean', type=float, nargs=3, default=[0.485, 0.456, 0.406], help='均值')
parser.add_argument('--std', type=float, nargs=3, default=[0.229, 0.224, 0.225], help='标准差')
parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'y'], help='颜色空间 (rgb 或 y)')
parser.add_argument('--lq_folder', type=str, default='Data/Datasets/RealSR', help='LQ 图像文件夹路径')
# 解析命令行参数
args = parser.parse_args()


# 将命令行参数转换为字典
opt = {
    'dataroot_lq': args.dataroot_lq,
    'meta_info_file': args.meta_info_file,
    'io_backend': {
        'type': args.io_backend_type,
        'db_paths': [args.dataroot_lq],
        'client_keys': ['lq']
    },
    'mean': args.mean,
    'std': args.std,
    'color': args.color,  # 动态选择颜色空间
}
opt = {
    'scale': 4,
    'train': {'lr': 0.001},
    'pretrained_g_path': 'Weights/DDG_Encoder.pth',
    'pretrained_e_path': 'Weights/PretrainEncoder.pth',
    'pretrained_g2_path': 'Weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4 (2).pth'
}




args = parser.parse_args()

assert len(args.patch_size) == 2, 'crop-size argument must contain 2 values'
args.patch_size = tuple(args.patch_size)
args.directory = f'{args.model}_{args.dataset}'
args.exp_name = f'{args.patch_size[0]}_{args.patch_size[1]}_{args.lr}_{args.lr_scheduler}_{args.loss_type}_{args.exp_name}'
args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]

