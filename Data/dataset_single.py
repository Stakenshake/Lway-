from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from Data.data_utils import paths_from_lmdb
from Utils.utils import FileClient, imfrombytes, img2tensor
from Utils.col_utils import rgb2ycbcr
from Utils.scandir import scandir

class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else [0.485, 0.456, 0.406]  # 默认的RGB均值（ImageNet）
        self.std = opt['std'] if 'std' in opt else [0.229, 0.224, 0.225]  # 默认的RGB标准差（ImageNet）

        self.lq_folder = 'Data/Datasets/RealSR'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

        # 新增的参数
        self.patch_size = (64, 64)  # 每个小 patch 的大小
        self.stride = 32  # 步幅，控制重叠程度

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # 将图像裁切成多个小的 patch
        lq_patches = self.get_patches(img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        lq_patches = [img2tensor(patch, bgr2rgb=True, float32=True) for patch in lq_patches]

        # normalize
        if self.mean is not None or self.std is not None:
            for patch in lq_patches:
                normalize(patch, self.mean, self.std, inplace=True)

        # 返回多个 patch
        return {'lq': lq_patches, 'lq_path': lq_path}

    def get_patches(self, image):
        """将图像分割成多个小 patch"""
        img_height, img_width = image.shape[:2]
        patches = []
        for i in range(0, img_height - self.patch_size[0] + 1, self.stride):
            for j in range(0, img_width - self.patch_size[1] + 1, self.stride):
                patch = image[i:i + self.patch_size[0], j:j + self.patch_size[1]]
                patches.append(patch)
        return patches

    def __len__(self):
        return len(self.paths)
