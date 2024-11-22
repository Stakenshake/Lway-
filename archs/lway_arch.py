import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)



class StyleGAN2_mod(nn.Module):

    def __init__(self, out_channels, modul_channels, kernel_size=3, noise=True):
        super().__init__()

        print('Modulation Method: StyleGAN2_mod')

        # self.conv = ResBlock(in_channels, out_channels)

        # generate global conv weights
        fan_in = out_channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.out_channels = out_channels
        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(modul_channels, self.out_channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, self.out_channels, self.out_channels, kernel_size, kernel_size)
        )
        self.conv_last = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        self.noise = noise
        if self.noise:
            self.weight_noise = nn.Parameter(torch.zeros(1))  # for noise injection

        self.activation = nn.LeakyReLU(0.1, True)

        print('Add random noise layer: ', self.noise)

    def forward(self, x, xg):
        # for global adjustation
        B, _, H, W = x.size()

        C = self.out_channels
        if len(xg.shape) == 4:  # [B, embed_ch, 1, 1]
            pass
        elif len(xg.shape) == 2:  # [B, embed_ch]
            xg = xg.unsqueeze(2).unsqueeze(2)  # [B, embed_ch, 1, 1]

        style = self.modulation(xg).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )

        # x = self.conv(x)
        x = x.contiguous()
        x_input = x.view(1, B * C, H, W)
        x_global = F.conv2d(x_input, weight, padding=self.padding, groups=B)
        x_global = x_global.view(B, C, H, W)

        if self.noise:
            b, _, h, w = x_global.shape
            n = x_global.new_empty(b, 1, h, w).normal_()
            x_global += self.weight_noise * n

        x = self.conv_last(x_global)
        x = self.activation(x)

        return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_modulation(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, degradation_embed_dim=512):
        super(ResidualBlock_noBN_modulation, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #self.modulation_layer = GFM_modulation(degradation_embed_dim, nf, noise=True)
        #print(degradation_embed_dim)
        self.modulation_layer = StyleGAN2_mod(out_channels=nf, modul_channels=degradation_embed_dim, kernel_size=3, noise=True)

    def forward(self, x, embedding):
        #print(x.shape)
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        res = identity + out
        res = self.modulation_layer(res, embedding)
        return res




class MSRResNet_Head(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_c=3, out_c=3, nf=64, scale=1, require_modulation=False, degradation_embed_dim=512):
        super(MSRResNet_Head, self).__init__()
        self.upscale = scale
        self.require_modulation = require_modulation
        print('require modulation: ', self.require_modulation)

        self.conv_first = nn.Conv2d(in_c, nf, 3, 1, 1, bias=True)

        if self.require_modulation:
            self.basic_block1 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
            self.basic_block2 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
            self.basic_block3 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
            self.basic_block4 = ResidualBlock_noBN_modulation(nf=nf, degradation_embed_dim=degradation_embed_dim)
        else:
            self.basic_block1 = ResidualBlock_noBN(nf=nf)
            self.basic_block2 = ResidualBlock_noBN(nf=nf)
            self.basic_block3 = ResidualBlock_noBN(nf=nf)
            self.basic_block4 = ResidualBlock_noBN(nf=nf)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_c, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, embedding=None):
        out = self.lrelu(self.conv_first(x))

        if self.require_modulation:
            out = self.basic_block1(out, embedding)
            out = self.basic_block2(out, embedding)
            out = self.basic_block3(out, embedding)
            out = self.basic_block4(out, embedding)
        else:
            out = self.basic_block1(out)
            out = self.basic_block2(out)
            out = self.basic_block3(out)
            out = self.basic_block4(out)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))

        return out
##LR reconstruction
##------------------------------------------------------------------------------------------------------------
##Degradation Encoder

class MSRResNet_wGR_i_fea(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, rfea_layer='RB16'):
        super(MSRResNet_wGR_i_fea, self).__init__()
        print('Model: MSRResNet_wGR_i (return feature)')
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk1 = make_layer(basic_block, nb // 4)
        self.recon_trunk2 = make_layer(basic_block, nb // 4)
        self.recon_trunk3 = make_layer(basic_block, nb // 4)
        self.recon_trunk4 = make_layer(basic_block, nb // 4)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 1:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

        self.rfea_layer = rfea_layer
        print('Return feature layer: {}'.format(self.rfea_layer))

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        fea_out_Conv1 = fea.view(1, -1)
        out = self.recon_trunk1(fea)
        fea_out_RB4 = out  # .view(1,-1)
        out = self.recon_trunk2(out)
        fea_out_RB8 = out  # .view(1,-1)
        out = self.recon_trunk3(out)
        fea_out_RB12 = out  # .view(1,-1)
        out = self.recon_trunk4(out)
        fea_out_RB16 = out  # .view(1,-1)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            fea_out_UP1 = out.view(1, -1)
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base

        if self.rfea_layer == 'Conv1':
            return out, fea_out_Conv1
        elif self.rfea_layer == 'RB4':
            return out, fea_out_RB4
        elif self.rfea_layer == 'RB8':
            return out, fea_out_RB8
        elif self.rfea_layer == 'RB12':
            return out, fea_out_RB12
        elif self.rfea_layer == 'RB16':
            return out, fea_out_RB16
        elif self.rfea_layer == 'UP1':
            return out, fea_out_UP1
class DeepDegradationEncoder_v2(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_v2, self).__init__()

        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        # checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        if checkpoint is not None:
            print('pretrained_DDR_model: {}'.format(checkpoint))
            load_net = torch.load(checkpoint)
        else:
            print('No pretrained_DDR_model found!')
            exit()
        self.DDR_extractor.load_state_dict(load_net, strict=True)

        print('Load Deep Degradation Representation Extractor successfully!')

        for i in self.DDR_extractor.parameters():
            i.requires_grad = False

        self.conv1 = nn.Conv2d(nf, nf * 2, 4, 2, 0, bias=True)
        self.conv2 = nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=True)
        self.conv3 = nn.Conv2d(nf * 4, nf * 8, 4, 2, 0, bias=True)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(nf * 8, nf * 8, 1, bias=True)
        self.conv1x1_2 = nn.Conv2d(nf * 8, nf * 8, 1, bias=True)
        self.conv1x1_3 = nn.Conv2d(nf * 8, nf * 8, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)

        out = self.lrelu(self.conv1(DeepDegradationRepre))  # in: [B, 64, H, W]  out: [B, 128, H/2, W/2]
        out = self.lrelu(self.conv2(out))  # in: [B, 128, H/2, W/2]  out: [B, 256, H/4, W/4]
        out = self.lrelu(self.conv3(out))  # in: [B, 256, H/4, W/4]  out: [B, 512, H/8, W/8]
        out = self.adapool(out)  # in: [B, 512, H/8, W/8]  out: [B, 512, 1, 1]
        out = self.lrelu(self.conv1x1_1(out))  # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        out = self.lrelu(self.conv1x1_2(out))  # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        out = self.conv1x1_3(out)  # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]

        DDR_embedding = out

        return DDR_embedding
##-------------------------------------------------------------------------------------------------------------
#self-supervised backbone