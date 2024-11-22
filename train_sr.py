import torch
import torch.nn as nn
import os
import warnings
import wandb
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
from Metrics.ntire.model_summary import get_model_flops, get_model_params
from Utils.print_utils import print_log_message, print_info_message
from Utils.lr_scheduler_real import EMA, LR_Scheduler
from Utils.Saver import MySaver
from Utils.msic import AverageMeter, set_seed, delete_state_module
from Metrics.psnr_ssim import _calculate_psnr_ssim_niqe

'''Data'''

'''Ignore Warnings'''
warnings.filterwarnings("ignore")
'''Drop problem images'''
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.seterr(divide='ignore', invalid='ignore')
'''Wandb'''
os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "dryrun"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Trainer():
    def __init__(self, args):

        """----------------------------------------------- opt -----------------------------------------------"""
        self.args = args

        """----------------------------------------------- Seed ----------------------------------------------"""
        set_seed(self.args.seed)

        """---------------------------------------------- Device ---------------------------------------------"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        ''' ---------------------------------------------- Saver -------------------------------------------- '''
        self.saver = MySaver(directory=self.args.directory, exp_name=self.args.exp_name)

        """----------------------------------------------- Model ---------------------------------------------"""
        from Model.model_import import model_import
        self.G1 = model_import(self.args.model1, scale=4)
        self.G2 = model_import(self.args.model2, scale=4)
        with torch.no_grad():
            h, w = 720 // self.args.scale_factor, 1280 // self.args.scale_factor
            params1 = get_model_params(self.G1) / 10 ** 6
            flops1 = get_model_flops(self.G1, (3, h, w), False) / 10 ** 9
            params2 = get_model_params(self.G2) / 10 ** 6
            flops2 = get_model_flops(self.G2, (3, h, w), False) / 10 ** 9
            print_info_message(f'Model 1 Parameters:{params1:,}, Flops:{flops1:,} of input size:{h}x{w}')
            print_info_message(f'Model 2 Parameters:{params2:,}, Flops:{flops2:,} of input size:{h}x{w}')
        self.saver.save_configs(args, params2, flops2,  h, w)


        """----------------------------------------------- Loss ----------------------------------------------"""
        from Loss.basic_loss import FFTLoss, CharbonnierLoss
        from Loss.dwt_loss import WeightedHighFrequencyLoss
        from Loss.rec_loss import RecLoss

        self.loss1 = RecLoss()
        self.loss2 = WeightedHighFrequencyLoss()

        """--------------------------------------------- Optimizer -------------------------------------------"""
        self.optim_net1 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G1.parameters()), self.args.lr)
        self.optim_net2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G2.parameters()), self.args.lr)


        """----------------------------------------------- Wandb ---------------------------------------------"""
        if self.args.wandb:
            wandb.init(project=self.args.wandb_project, entity="")
            print_info_message('Use Wandb!')
        else:
            print_info_message('Not use Wandb!')

        ''' ----------------------------------------------- Cuda -------------------------------------------- '''
        if self.args.use_cuda:
            print_info_message('Use cuda!')
            self.G1 = nn.DataParallel(self.G1, device_ids=self.args.gpu_ids).to(self.device)
            self.G2 = nn.DataParallel(self.G2, device_ids=self.args.gpu_ids).to(self.device)
            self.loss1 = self.loss1.to(self.device)
            self.loss2 = self.loss2.to(self.device)

        """---------------------------------------------- Dataset --------------------------------------------"""
        from Data.dataset_paired import Dataloader
        from Data.dataset_single import SingleImageDataset

        # 构建 paired 数据集
        self.train_dataloader, self.set5_dataloader, self.set14_dataloader, self.urban100_dataloader, self.manga109_dataloader = \
            Dataloader(self.args.scale_factor, self.args.patch_size[0], self.args.train_batchsize,
                       self.args.val_batchsize,
                       self.args.workers, pin_memory=self.args.pin_memory)

        # 构建 single 数据集
        self.single_image_dataset = SingleImageDataset(args)
        self.single_image_dataloader = Dataloader(self.args.scale_factor, self.args.patch_size[0], self.args.train_batchsize,
                       self.args.val_batchsize,
                       self.args.workers, pin_memory=self.args.pin_memory)# 指定放大倍数

        # 打印 train 数据集信息
        self.train_dataloader_len = len(self.train_dataloader)
        print_info_message('Train:      Train_dataloader_len:{} | Train images:{}'.format(
            self.train_dataloader_len, self.args.train_batchsize * self.train_dataloader_len))
        print_info_message(
            f'Single Dataset: {len(self.single_image_dataset)} images loaded from {self.args.single_dataset_path}')
        print_info_message(f'Benchmark: set5:{len(self.set5_dataloader)} | '
                           f'set14:{len(self.set14_dataloader)} | '
                           f'urban100:{len(self.urban100_dataloader)} | '
                           f'manga109:{len(self.manga109_dataloader)}')


        """-------------------------------------------- LR_scheduler -----------------------------------------"""
        if self.args.lr_scheduler == 'poly' or 'cos' or 'step' or 'multistep':
            self.lr_scheduler_net = LR_Scheduler(self.args.lr_scheduler, base_lr=self.args.lr,
                                                 num_epochs=self.args.epochs, iters_per_epoch=self.train_dataloader_len,
                                                 lr_step=self.args.step_size, warmup_epochs=self.args.warmup_epochs,
                                                 step_gamma=self.args.step_gamma, eta_min=1e-7)
        elif self.args.lr_scheduler == 'reducelr':
            self.lr_scheduler_net1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim_net1, 'min', 0.5, 6)
            self.lr_scheduler_net2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim_net2, 'min', 0.5, 6)
        print_info_message('Use {} scheduler mode'.format(self.args.lr_scheduler))

        """---------------------------------------------- EMA ---------------------------------------------"""
        if self.args.ema:
            self.ema_g1 = EMA(self.G1, 0.999)
            self.ema_g2 = EMA(self.G2,0.999)
            self.ema_g1.register()
            self.ema_g2.register()
            print_info_message('Use EMA!')

        """------------------------------------------ Resume checkpoint --------------------------------------"""
        self.best_pred = 0.
        if self.args.resume is not None:
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.args.start_epoch = checkpoint['epoch'] + 1
            if self.args.delete_module:
                checkpoint = delete_state_module(checkpoint)
            ###
            model_dict1 = self.G1.module.state_dict()
            model_dict2 = self.G2.module.state_dict()
            overlap1 = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict1}
            model_dict1.update(overlap1)
            print_info_message(f'{(len(overlap1) * 1.0 / len(checkpoint["state_dict"]) * 100):.4f}% weights is loaded!')
            print_info_message(f'{(len(overlap1) * 1.0 / len(model_dict1) * 100):.4f}% params is init!')
            print_info_message(f'Drop Keys: {[k for k, v in checkpoint["state_dict"].items() if k not in model_dict1]}')
            ###
            self.G2.module.load_state_dict(model_dict2)
            overlap2 = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict2}
            model_dict2.update(overlap2)
            print_info_message(f'{(len(overlap2) * 1.0 / len(checkpoint["state_dict"]) * 100):.4f}% weights is loaded!')
            print_info_message(f'{(len(overlap2) * 1.0 / len(model_dict2) * 100):.4f}% params is init!')
            print_info_message(f'Drop Keys: {[k for k, v in checkpoint["state_dict"].items() if k not in model_dict2]}')
            ###
            self.G2.module.load_state_dict(model_dict2)

            if not self.args.finetune:
                self.optim_net1.load_state_dict(checkpoint['optimizer'])
                self.optim_net2.load_state_dict(checkpoint['optimizer'])####???
                self.best_pred = checkpoint['best_pred']
                print_info_message('Not Finetune!')
            elif self.args.finetune:
                self.args.start_epoch = 0
                self.best_pred = 0.0
                print_info_message(f'Finetune!')

            print_info_message(f"Loading checkpoint:'{self.args.resume}' epoch:{checkpoint['epoch']} previous_best_pred:{checkpoint['best_pred']}")

    def net1_train(self, now_epoch):
        self.G1.train()
        train_losses = AverageMeter()
        tbar = tqdm(self.train_dataloader, unit='image')
        for i, (im, gt) in enumerate(tbar):
            self.lr_scheduler_net(self.optim_net1, i, now_epoch, self.best_pred)
            if i == 0 and now_epoch == 0: print(im.shape, gt.shape)
            im, gt = im.to(self.device), gt.to(self.device)

            im_rec = self.G1(im, gt)
            loss1 = self.loss1(im_rec, im)  # l1+lpips

            tbar.set_description(f'[Train Epoch:{now_epoch}] [Total_loss:{loss1.item():.4f}]')
            train_losses.update(loss1.item(), n=im.shape[0])
            if self.args.wandb:wandb.log({"train_iter_loss": loss1.item()})

            self.optim_net1.zero_grad()
            loss1.backward()
            self.optim_net1.step()

            if self.args.ema:
                self.ema_g1.update()

            if self.args.iter_eval:
                if i % self.args.eval_freq == 0 or i == -1:
                    self.Benchmark_iter(now_epoch, i)

        # epoch / iter mode test and save
        if self.args.epoch_eval:
            self.Benchmark(now_epoch)

        print_info_message('Epoch:{}, numImages:{}, Train_loss:{:.3f}'.format(epoch, i * self.args.train_batchsize + im.data.shape[0], train_losses.avg))
        if self.args.wandb: wandb.log({"total_train_loss": train_losses.avg})
        self.saver.save_record_train(now_epoch, train_losses.avg)

    def net2_train(self, now_epoch2):
        self.G1.freeze_network()
        self.G2.freeze_layers(transformer=True)
        self.G2.train()
        train_losses = AverageMeter()
        tbar = tqdm(self.single_image_dataloader, unit='image')
        for i, (im) in enumerate(tbar):
            self.lr_scheduler_net(self.optim_net2, i, now_epoch2, self.best_pred)
            if i == 0 and now_epoch2 == 0: print(im.shape)
            im, gt = im.to(self.device)

            g2_hq = self.G2(im)
            g2_hq_rec = self.G1(im, g2_hq)
            loss2 = self.loss2(g2_hq_rec, im)  # l1+lpips


            tbar.set_description(f'[Total_loss:{loss2.item():.4f}]')
            train_losses.update(loss2.item(), n=im.shape[0])
            if self.args.wandb : wandb.log({"train_iter_loss": loss2.item()})

            self.optim_net2.zero_grad()
            loss2.backward()
            self.optim_net2.step()

            if self.args.ema:
                self.ema_g2.update()

            if self.args.iter_eval:
                if i % self.args.eval_freq == 0 or i == -1:
                    self.Benchmark_iter(now_epoch2, i)

        # epoch / iter mode test and save
        if self.args.epoch_eval:
            self.Benchmark(now_epoch2)

        print_info_message(
            'Epoch:{}, numImages:{}, Train_loss:{:.3f}'.format(epoch, i * self.args.train_batchsize + im.data.shape[0],
                                                               train_losses.avg))
        if self.args.wandb: wandb.log({"total_train_loss": train_losses.avg})
        self.saver.save_record_train(now_epoch2, train_losses.avg)

    @torch.no_grad()
    def Benchmark(self, now_epoch):
        if self.args.ema:
            self.ema_g2.apply_shadow()

        self.G2.eval()
        lr_rate = self.optim_net2.param_groups[0]['lr']
        val_losses, psnr, ssim, niqe = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        tbar = tqdm(self.set5_dataloader, unit='image')
        with torch.no_grad():
            for i, (im, gt) in enumerate(tbar):
                if i == 0 and now_epoch == 0: print(im.shape, gt.shape)
                # save input
                # self.saver.save_sr(i, 500, gt, now_epoch, denormal=False, flag='hr')
                # self.saver.save_sr(i, 500, im, now_epoch, denormal=False, flag='lr')
                im, gt = im.to(self.device), gt.to(self.device)
                output = self.G2(im)
                # save sr
                # self.saver.save_sr(i, 500, output, now_epoch, denormal=False, flag='sr')
                loss = self.loss2(output, gt)

                val_losses.update(loss.item(), n=im.shape[0])
                tbar.set_description(f'[Val Epoch:{now_epoch}] [val_loss:{loss.item():.4f}]')
                if self.args.wandb: wandb.log({"val_iter_loss": loss.item()})

                psnr_temp, ssim_temp, niqe_temp, batch = _calculate_psnr_ssim_niqe(output, gt, crop_border=0,
                                                                                   input_order='CHW',
                                                                                   test_y_channel=True, mean=(0, 0, 0),
                                                                                   std=(1, 1, 1))
                psnr.update(psnr_temp, batch)
                ssim.update(ssim_temp, batch)
                niqe.update(niqe_temp, batch)

            avg_psnr = psnr.avg
            avg_ssim = ssim.avg
            avg_niqe = niqe.avg
            if self.args.wandb:
                wandb.log({"total_val_loss": val_losses.avg,
                           "Avg_PSNR": avg_psnr,
                           "Avg_SSIM": avg_ssim,
                           "Avg_niqe": avg_niqe})
            self.saver.save_record_val(now_epoch, lr_rate, val_losses.avg, avg_psnr, avg_ssim, avg_niqe)
            print_log_message(
                f'val loss_script:{val_losses.avg:.4f}  Avg_psnr:{avg_psnr:.6f} Avg_ssim:{avg_ssim:.6f} Avg_niqe:{avg_niqe:.6f}')

        new_pred = avg_psnr
        self.saver.save_checkpoint_override(
            {
                'epoch': now_epoch,
                'state_dict': self.G2.module.state_dict(),  #
                'optimizer': self.optim_net2.state_dict(),
                'pred': new_pred,
                'ssim': avg_ssim,
                'niqe': avg_niqe
            }, now_epoch, new_pred, self.best_pred, self.args, print_=True
        )
        if new_pred > self.best_pred:
            self.best_pred = new_pred

        if self.args.ema:
            self.ema_g2.restore()

    @torch.no_grad()
    def Benchmark_iter(self, now_epoch, iter):
        if self.args.ema:
            self.ema_g2.apply_shadow()

        self.G2.eval()
        lr_rate = self.optim_net2.param_groups[0]['lr']
        val_losses, psnr, ssim, niqe = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        with torch.no_grad():
            for i, (im, gt) in enumerate(self.set5_dataloader):
                # if i == 0 and now_epoch == 0: print(im.shape, gt.shape)
                # save input
                # self.saver.save_sr(i, 500, gt, now_epoch, denormal=False, flag='hr')
                # self.saver.save_sr(i, 500, im, now_epoch, denormal=False, flag='lr')
                im, gt = im.to(self.device), gt.to(self.device)
                output = self.G2(im)
                # save sr
                # self.saver.save_sr(i, 500, output, now_epoch, denormal=False, flag='sr')
                # loss = self.l1(output, gt)

                # val_losses.update(loss.item(), n=im.shape[0])
                # tbar.set_description(f'[Val Epoch:{now_epoch}] [val_loss:{loss.item():.4f}]')
                # if self.args.wandb: wandb.log({"val_iter_loss": loss.item()})

                psnr_temp, ssim_temp, niqe_temp, batch = _calculate_psnr_ssim_niqe(output, gt, crop_border=0,
                                                                                   input_order='CHW',
                                                                                   test_y_channel=True, mean=(0, 0, 0),
                                                                                   std=(1, 1, 1))
                psnr.update(psnr_temp, batch)
                ssim.update(ssim_temp, batch)
                niqe.update(niqe_temp, batch)

            avg_psnr = psnr.avg
            avg_ssim = ssim.avg
            avg_niqe = niqe.avg
            if self.args.wandb:
                wandb.log({"total_val_loss": val_losses.avg,
                           "Avg_PSNR": avg_psnr,
                           "Avg_SSIM": avg_ssim,
                           "Avg_niqe": avg_niqe})
            self.saver.save_record_val(now_epoch, lr_rate, val_losses.avg, avg_psnr, avg_ssim, avg_niqe)

        new_pred = avg_psnr
        self.saver.save_checkpoint_override(
            {
                'epoch': now_epoch,
                'iter': iter,
                'state_dict': self.G2.module.state_dict(),  #
                'optimizer': self.optim_net2.state_dict(),
                'pred': new_pred,
                'ssim': avg_ssim,
                'niqe': avg_niqe
            }, now_epoch, new_pred, self.best_pred, self.args, print_=False
        )
        if new_pred > self.best_pred:
            self.best_pred = new_pred

        if self.args.ema:
            self.ema_g2.restore()


if __name__ == '__main__':
    from option import args

    trainer = Trainer(args)

    for epoch in range(trainer.args.start_epoch1, args.epochs1):
        trainer.net1_train(epoch)
    for epoch in range(trainer.args.start_epoch2, args.epochs2):
        trainer.net2_train(epoch)
    ##print(f'The last best predict is: {trainer.best_pred}')
