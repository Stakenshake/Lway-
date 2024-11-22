import torch
import torch.optim as optim
from archs.lway_arch import DeepDegradationEncoder_v2 , MSRResNet_Head
from .base_model import BaseModel
from Loss.rec_loss import RecLoss
from Loss.dwt_loss import WeightedHighFrequencyLoss
from archs.swinir_arch import SwinIR



# 第一个模型：传统超分辨率模型
class LRReconstructionModel(BaseModel):
    def __init__(self, opt):
        super(LRReconstructionModel, self).__init__(opt)

        # 使用经典的超分辨率网络架构，例如RRDBNet
        self.g = MSRResNet_Head(
            in_c=3,  # 输入通道数
            out_c=3,  # 输出通道数
            nf=64,  # 基础通道数
            scale=opt['scale'],  # 放大倍数（2x, 3x, 4x）
            require_modulation=True,  # 是否使用调制
            degradation_embed_dim=512  # 退化嵌入维度
        )
        self.e = DeepDegradationEncoder_v2(in_nc=3, out_nc=512, nf=64, nb=16, upscale=4, checkpoint=opt.get('checkpoint', None))
        if opt.get('pretrained_g_path', None):  # 如果给定了预训练权重路径
            self.load_pretrained_weights(self.g, opt['pretrained_g_path'])

        if opt.get('pretrained_e_path', None):  # 如果给定了预训练编码器权重路径
            self.load_pretrained_weights(self.e, opt['pretrained_e_path'])
        # 损失函数：这里使用L1损失
        self.loss_fns = {
            'L_sum': RecLoss(),
        }

        # 优化器
        self.optimizer_g = optim.Adam(self.g.parameters(), lr=opt['train']['lr'], betas=(0.9, 0.99))
        self.optimizer_e = optim.Adam(self.e.parameters(), lr=opt['train']['lr'], betas=(0.9, 0.99))

        # 网络初始化
        self.print_network()

        def load_pretrained_weights(self, model, weight_path):
            """
            加载预训练权重
            """
            print(f"加载预训练权重: {weight_path}")
            pretrained_dict = torch.load(weight_path)  # 加载权重文件
            model_dict = model.state_dict()  # 获取当前模型的权重字典

            # 过滤掉模型字典中不需要的参数
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 更新当前模型的权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        def freeze_layers(self, freeze_layers):
            """
            冻结模型的部分层参数
            freeze_layers: str, 'all' 或 'conv' 或 'transformer'
                - 'all'：冻结所有层的参数
                - 'conv'：只冻结卷积层的参数
                - 'transformer'：只冻结 Transformer 层的参数
            """
            if freeze_layers == 'all':
                for param in self.net_g.parameters():
                    param.requires_grad = False
            elif freeze_layers == 'conv':
                # 冻结卷积层参数
                for name, param in self.net_g.named_parameters():
                    if 'conv' in name:
                        param.requires_grad = False
            elif freeze_layers == 'transformer':
                # 冻结 Transformer 层的参数
                for name, param in self.net_g.named_parameters():
                    if 'layers' in name:  # 假设 Transformer 层的名称包含 'layers'
                        param.requires_grad = False

    def forward(self, lr_img , hr_img):
        # 正向传播：生成高分辨率图像
        ddr_embedding = self.e(lr_img)
        lr_img_rec = self.g(hr_img, ddr_embedding)

        return lr_img_rec

    def optimize_parameters(self, data=None):
        """计算损失并优化网络"""
        lr_img, hr_img = data  # 从输入数据中提取低分辨率和高分辨率图像
        lr_img_rec = self.forward(lr_img, hr_img)
        loss = self.loss_fns['L_sum'](lr_img, lr_img_rec)

        # 优化网络
        self.optimizer_g.zero_grad()
        self.optimizer_e.zero_grad()
        loss.backward()
        self.optimizer_g.step()
        self.optimizer_e.step()
        return loss

    def freeze_network(self):
        """冻结生成器和编码器的参数"""
        for param in self.g.parameters():
            param.requires_grad = False
        for param in self.e.parameters():
            param.requires_grad = False




# 第二个模型：自监督学习模型
class SelfSupervisedModel(BaseModel):
    def __init__(self, opt, sr_model):
        super(SelfSupervisedModel, self).__init__(opt)

        self.net_g = SwinIR(
            img_size=64,  # 输入图像尺寸
            patch_size=1,  # Patch 大小
            in_chans=3,  # 输入图像通道数
            embed_dim=96,  # 嵌入维度
            depths=(6, 6, 6, 6),  # 每层 Transformer 的深度
            num_heads=(6, 6, 6, 6),  # 注意力头数
            window_size=7,  # 窗口大小
            mlp_ratio=4.0,  # MLP 层的隐藏维度与嵌入维度的比率
            upscale=2,  # 超分辨率倍数
            upsampler='pixelshuffle',  # 上采样方式
            resi_connection='1conv'  # 残差连接的卷积层数
        )
        self.loss_fns = {
            'L_sum': WeightedHighFrequencyLoss(),
        }

        # 优化器
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=opt['train']['lr'], betas=(0.9, 0.99))



        # 冻结SR模型的参数
        self.sr_model = sr_model  # 引入第一个模型（SR模型）
        self.sr_model.freeze_network()  # 让第一个模型的参数不可训练

        # 网络初始化
        self.print_network()
        if opt.get('pretrained_g2_path', None):
            self.load_pretrained_weights(self.net_g, opt['pretrained_g2_path'])

    def load_pretrained_weights(self, model, weight_path):
        """
        加载预训练权重
        """
        print(f"加载预训练权重: {weight_path}")
        pretrained_dict = torch.load(weight_path)  # 加载权重文件
        model_dict = model.state_dict()  # 获取当前模型的权重字典

        # 过滤掉模型字典中不需要的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 更新当前模型的权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def forward(self, lr_img):
        # 生成高分辨率图像
        sr_img = self.net_g(lr_img)
        return sr_img

    def optimize_parameters(self, lr_img, hr_img=None, need_backward=True):
        """
        自监督训练的优化步骤

        Args:
            lr_img (torch.Tensor): 低分辨率图像。
            hr_img (torch.Tensor, optional): 高分辨率图像。默认为None，适用于自监督学习。
            need_backward (bool): 是否需要执行反向传播，默认为True。

        Returns:
            loss (torch.Tensor): 损失值
        """
        # 自监督：我们不需要高分辨率图像作为目标，而是通过生成的超分辨率图像自我监督
        sr_img = self.forward(lr_img)
        lr_img_rec = self.sr_model.forward(lr_img, sr_img)

        # 自监督学习的损失：L1损失
        loss = self.loss_fns['L_sum'](lr_img_rec, lr_img)

        # 只有在需要反向传播时才进行反向传播
        if need_backward:
            self.optimizer_g.zero_grad()
            loss.backward()
            self.optimizer_g.step()

        return loss

    def feed_data(self, data):
        """
        处理输入的数据，返回低分辨率图像（lr_img）并将其传递给训练过程
        这里假设 data 是一个字典，包含需要的 LR 图像。
        """
        lr_img = data['lr']  # 从输入数据中提取低分辨率图像
        lr_img = lr_img.cuda() if torch.cuda.is_available() else lr_img  # 处理GPU/CPU
        return lr_img