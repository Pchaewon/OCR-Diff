#--- main + SR ---#
import os
import cv2
import sys
import time
import math
import copy
import torch
import pickle
import shutil
import random
import logging
import functools
from inspect import isfunction
from functools import partial
import numpy as np
import torchvision
from einops import rearrange
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from math import exp
from IPython import embed
from datetime import datetime
from time import gmtime, strftime
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter

from interfaces import imresize
from utils import utils_moran , utils_cdist
from utils.util import str_filt 
from utils import util, ssim_psnr 
from utils.meters import AverageMeter 
from utils.metrics import get_str_list, Accuracy
from torchvision.transforms import ToTensor

import string
import torchvision
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from IPython import embed
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from model import  crnn , recognizer, bicubic

import dataset.SRDiff_dataset as dataset
from dataset.SRDiff_dataset import alignCollate_real, ConcatDataset, lmdbDataset_real, alignCollate_syn

from loss import gradient_loss, percptual_loss, tp_loss

from utils import util, ssim_psnr, utils_moran, utils_crnn
from utils.labelmaps import get_vocabulary, labels2strs


os.environ["CUDA_VISIBLE_DEVICES"] = "2, 5"

device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- get parameter --- #
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    return total_num

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb) 
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
# Mish 
'''
ReLU보다 gradient가 smoothing함.
'''
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
# ResnetBlock
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        if groups == 0:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                Mish()
            )
        else:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                nn.GroupNorm(groups, dim_out),
                Mish()
            )
    
    def forward(self, x):
        # i=0
        # conv_layer = self.block[1]
        # print(i,"================",conv_layer.weight.shape)
        # i+=1
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = 0, groups=8):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(
                Mish(),
                nn.Linear(time_emb_dim, dim_out)
            )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
       
        if time_emb is not None:

            # print('Before Reshape t shape:', time_emb.shape)
            # print('After Reshape t shape:', self.mlp(time_emb)[:, :, None, None].shape)
            h+= self.mlp(time_emb)[:, :, None, None]
            
            # print('h shape:', h.shape)
        
        if cond is not None:
            h += cond

        h = self.block2(h)

        return h + self.res_conv(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return self.fn(x) * self.g
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q,k,v =rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim = -1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    
# U-Net
def initialize_weights(net_l, scale=0.1):
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

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1,2,3,4), cond_dim=32):
        super().__init__()
        dims = [3, *map(lambda m: dim * m, dim_mults)] # 
        in_out = list(zip(dims[:-1],dims[1:]))
        # in_out =[]
        # in_out.append(tuple(dims))
        groups = 0

        self.res = True
        self.use_attn = True
        self.use_wn = False
        self.weight_init = False
        self.up_input = True#False
        rrdb_num_block = 8
        sr_scale = 2

        # self.cond_proj = nn.ConvTranspose2d(32 * ((8+1)//3), 512, kernel_size=4, stride=2, padding=1)
        self.cond_proj = nn.ConvTranspose2d(cond_dim * ((rrdb_num_block+1)//3), 
                                            dim, sr_scale*2, sr_scale, sr_scale//2)
        self.time_pos_emb = SinusoidalPosEmb(dim) #transformer positional encoding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            Mish(),
            nn.Linear(dim*4, dim)
        )
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Downsample block
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        
        if self.use_attn : 
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        # Upsample block
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out*2, dim_in, time_emb_dim = dim, groups = groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

        self.up_proj = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(3, dim, 3)
        )

        if self.use_wn:
            self.apply_weight_norm()
        if self.weight_init:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
    
    def forward(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time) # timestep embedding t_{e}

        t = self.mlp(t)
        # print('Before t shape:', t.shape)


        h = []

        # print('cond:',cond)
        # print('cond length:',len(cond))
        cond = self.cond_proj(torch.cat(cond[2::3], 1))
        # print('cond shape2:', cond.shape)

        # downsampling
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):

            # print('After t shape:', t.shape)
            x = resnet(x, t)
            x = resnet2(x, t)

            if i == 0:
                # print('cond 3:',cond.shape)
                x = x + cond 
                # print('cond 4:',x.shape)

            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsampling
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)
    
    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)

# RRBDNet
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
    
class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

def make_layer(block, n_layers, seq=False):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if seq:
         return nn.Sequential(*layers)
    else:
        return nn.Sequential(*layers)
    
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # if hparams['sr_scale'] == 8:
        #     self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2
        # print('RRDB x:', x.shape)
        fea_first = fea = self.conv_first(x)
        # print('RRDB fea_first:', fea_first.shape)

        for l in self.RRDB_trunk:
            fea = l(fea)
            # print('RRDB fea:', fea.shape)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        # print('RRDB trunk:', trunk.shape)
        fea = fea_first + trunk
        # print('RRDB fea > fea_first+:', fea.shape)
        feas.append(fea)

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print('RRDB fea1:', fea.shape)
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print('RRDB fea2:', fea.shape)
        # if hparams['sr_scale'] == 8:
        #     fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea_hr = self.HRconv(fea)
        # print('RRDB fea_hr:', fea_hr.shape)
        out = self.conv_last(self.lrelu(fea_hr))
        # print('RRDB out:', out.shape)
        out = out.clamp(0, 1)
        # print('RRDB out1:', out.shape)
        out = out * 2 - 1
        # print('RRDB out2:', out.shape)
        if get_fea:
            return out, feas
        else:
            return out
        
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        img1 = img1 * 0.5 + 0.5
        img2 = img2 * 0.5 + 0.5
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# diffusion model
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, rrdb_net, timesteps=3000, loss_type='l1'):
        super().__init__()
        self.denoise_fn = denoise_fn
        # condition net
        self.rrdb = rrdb_net
        self.ssim_loss = SSIM(window_size=11)
        
        self.use_rrdb = True
        self.fix_rrdb = True
        self.res = True
        self.clip_input = True
        self.res_rescale = 2.0
        self.aux_l1_loss = True
        self.aux_ssim_loss = False
        self.aux_percep_loss = False

        betas = cosine_beta_schedule(timesteps, s=0.008)
        # if hparams['aux_percep_loss']:
        #     self.percep_loss_fn = [PerceptualLoss()]
        # if hparams['beta_schedule'] == 'cosine':
        #     betas = cosine_beta_schedule(timesteps, s=hparams['beta_s'])
        # if hparams['beta_schedule'] == 'linear':
        #     betas = get_beta_schedule(timesteps, beta_end=hparams['beta_end'])
        #     if hparams['res']:
        #         betas[-1] = 0.999

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0) # 누적 곱
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])


        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.sample_tqdm = True


    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise_pred, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def forward(self, img_hr, img_lr, img_lr_up, t=None, *args, **kwargs):
        x = img_hr
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() \
            if t is None else torch.LongTensor([t]).repeat(b).to(device)
        if self.use_rrdb:
            if self.fix_rrdb:
                self.rrdb.eval()
                with torch.no_grad():
                    rrdb_out, cond = self.rrdb(img_lr, True)
            else:
                rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr
        x = self.img2res(x, img_lr_up)
        p_losses, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(x, t, cond, img_lr_up, *args, **kwargs)
        
        print('noise predict shape:', noise_pred.shape)
        ret = {'q': p_losses}
        if not self.fix_rrdb:
            if self.aux_l1_loss:
                ret['aux_l1'] = F.l1_loss(rrdb_out, img_hr)
            if self.aux_ssim_loss:
                ret['aux_ssim'] = 1 - self.ssim_loss(rrdb_out, img_hr)
            if self.aux_percep_loss:
                ret['aux_percep'] = self.percep_loss_fn[0](img_hr, rrdb_out)

        # x_recon = self.res2img(x_recon, img_lr_up)
        x_tp1 = self.res2img(x_tp1, img_lr_up)
        x_t = self.res2img(x_t, img_lr_up)
        x_t_gt = self.res2img(x_t_gt, img_lr_up)
        return ret, (x_tp1, x_t_gt, x_t), t
    

    def p_losses(self, x_start, t, cond, img_lr_up, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_gt = self.q_sample(x_start=x_start, t=t - 1, noise=noise)

        print('noise shape:',x_t_gt.shape)

        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, img_lr_up)
        x_t_pred, x0_pred = self.p_sample(x_tp1_gt, t, cond, img_lr_up, noise_pred=noise_pred)

        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == 'ssim':
            loss = (noise - noise_pred).abs().mean()
            loss = loss + (1 - self.ssim_loss(noise, noise_pred))
        else:
            raise NotImplementedError()
        return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred 

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t_cond = (t[:, None, None, None] >= 0).float()
        t = t.clamp_min(0)
        return (
                       extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                       extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               ) * t_cond + x_start * (1 - t_cond)
    
    @torch.no_grad()
    def p_sample(self, x, t, cond, img_lr_up, noise_pred=None, clip_denoised=True, repeat_noise=False):
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond, img_lr_up=img_lr_up)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred


    @torch.no_grad()
    def sample(self, img_lr, img_lr_up, shape, save_intermediate=False):
        device = self.betas.device
        b = shape[0]
        if not self.res:
            t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
            img = self.q_sample(img_lr_up, t)
        else:
            img = torch.randn(shape, device=device)
        if self.use_rrdb:
            rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr
        it = reversed(range(0, self.num_timesteps))
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        images = []
        for i in it:
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up)
            if save_intermediate:
                img_ = self.res2img(img, img_lr_up)
                x_recon_ = self.res2img(x_recon, img_lr_up)
                images.append((img_.cpu(), x_recon_.cpu()))
        img = self.res2img(img, img_lr_up)
        if save_intermediate:
            return img, rrdb_out, images
        else:
            return img, rrdb_out
        

    @torch.no_grad()
    def interpolate(self, x1, x2, img_lr, img_lr_up, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        if self.use_rrdb:
            rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            cond = img_lr

        assert x1.shape == x2.shape

        x1 = self.img2res(x1, img_lr_up)
        x2 = self.img2res(x2, img_lr_up)

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up)

        img = self.res2img(img, img_lr_up)
        return img
    
    def res2img(self, img_, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = self.clip_input
        if self.res:
            if clip_input:
                img_ = img_.clamp(-1, 1)
            
            # print('res2img:', img_.shape)
            img_ = img_ / self.res_rescale + img_lr_up
            # print('After res2img:', img_.shape)
        return img_

    def img2res(self, x, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = self.clip_input
        if self.res:
            x = (x - img_lr_up) * self.res_rescale
            if clip_input:
                x = x.clamp(-1, 1)
        return x

align_collate = alignCollate_real
load_dataset = lmdbDataset_real

from torch.autograd import Variable

alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'

def get_alphabet_len():
    return len(alphabet)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, attention_map=None):
    "Compute 'Scaled Dot Product Attention'"

    if attention_map is not None:
        return torch.matmul(attention_map, value), attention_map

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, attention_map=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, attention_map=attention_map)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), attention_map


class ResNet(nn.Module):
    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        # x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        # x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        # x = self.layer4_pool(x)
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return F.softmax(self.proj(x))
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        # print("embed",embed)
        # embed = self.lut(x)
        # print(embed.requires_grad)
        return embed


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        self.multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1, compress_attention=True)
        self.mul_layernorm2 = LayerNorm(features=1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

    def forward(self, text, conv_feature, attention_map=None):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()

        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None, attention_map=attention_map)
        result = self.mul_layernorm2(result + word_image_align)
        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = ResNet(num_in=1, block=BasicBlock, layers=[1, 2, 5, 3])

    def forward(self, input):
        conv_result = self.cnn(input)
        return conv_result


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        word_n_class = get_alphabet_len()
        self.embedding_word = Embeddings(512, word_n_class)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=5000)

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generator_word = Generator(1024, word_n_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, text_length, text_input, test=False, attention_map=None):

        conv_feature = self.encoder(image) # batch, 1024, 8, 32
        text_embedding = self.embedding_word(text_input) # batch, text_max_length, 512
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda() # batch, text_max_length, 512
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2) # batch, text_max_length, 1024
        batch, seq_len, _ = text_input_with_pe.shape

        text_input_with_pe, word_attention_map = self.decoder(text_input_with_pe, conv_feature, attention_map=attention_map)
        word_decoder_result = self.generator_word(text_input_with_pe)
        total_length = torch.sum(text_length).data
        probs_res = torch.zeros(total_length, get_alphabet_len()).type_as(word_decoder_result.data)

        start = 0
        for index, length in enumerate(text_length):
            # print("index, length", index,length)
            length = length.data
            probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
            start = start + length

        if not test:
            return probs_res, word_attention_map, None
        else:
            # return word_decoder_result, word_attention_map, text_input_with_pe
            return word_decoder_result

from loss.weight_ce_loss import weight_cross_entropy


def to_gray_tensor(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 1:2, :, :]
    B = tensor[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


def str_filt(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all': string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    str_ = str_.lower()
    return str_


class TPLoss(nn.Module):
    def __init__(self, args):
        super(TPLoss, self).__init__()
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.english_dict = {}
        for index in range(len(self.english_alphabet)):
            self.english_dict[self.english_alphabet[index]] = index

        self.build_up_transformer()

    def build_up_transformer(self):

        transformer = Transformer().cuda()
        transformer = nn.DataParallel(transformer)
        transformer.load_state_dict(torch.load('./dataset/mydata/pretrain_transformer.pth'))
        transformer.eval()
        self.transformer = transformer

    def label_encoder(self, label):
        batch = len(label)

        length = [len(i) for i in label]
        length_tensor = torch.Tensor(length).long().cuda()

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = torch.Tensor(text_gt).long().cuda()

        input_tensor = torch.from_numpy(input_tensor).long().cuda()
        return length_tensor, input_tensor, text_gt


    def forward(self,sr_img, hr_img, label):

        # mse_loss = self.mse_loss(sr_img, hr_img)
        label = [str_filt(i, 'lower')+'-' for i in label]
        length_tensor, input_tensor, text_gt = self.label_encoder(label)
        # print('length_tensor:',length_tensor)
        # print('length tensor shape:', length_tensor.shape)
        # print('input_tensor:',input_tensor)
        # print('input_tensor shape:',input_tensor.shape)
         # print('text_gt shape:',text_gt.shape)
        hr_pred, word_attention_map_gt, hr_correct_list = self.transformer(to_gray_tensor(hr_img), length_tensor,
                                                                          input_tensor)
        sr_pred, word_attention_map_pred, sr_correct_list = self.transformer(to_gray_tensor(sr_img), length_tensor,
                                                                            input_tensor)
        #attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
        # recognition_loss = self.l1_loss(hr_pred, sr_pred)
        # print('sr_pred:', sr_pred.shape)
         # print('text_gt:', text_gt.shape)
        recognition_loss = weight_cross_entropy(sr_pred, text_gt)

        return recognition_loss

        

# ---TextBase --- #
class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
     
        # -- Dataset
        self.align_collate = alignCollate_real # images_HR, images_lr, label_strs
        self.load_dataset = lmdbDataset_real
        
        # -- parameters
        self.resume = self.config.TRAIN.resume
        self.scale_factor = self.config.TRAIN.down_sample_scale
        self.batch_size = self.config.TRAIN.batch_size
        self.test_data_dir = self.args.test_data_dir
        self.mask = self.args.mask
        self.voc_type = self.config.TRAIN.voc_type # all
        self.max_len = config.TRAIN.max_len # 100
        
        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }

        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len

        # -- CUDA setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # -- metrics
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)
        
        if not args.test and not args.demo:
            self.clean_old_ckpt()
        # -- event logging        
        self.logging = logging
        self.make_logger()
        self.make_writer()
    
    # -- make_logger
    def make_logger(self):
        self.logging.basicConfig(filename="checkpoint/{}/log.txt".format(self.args.exp_name),
                            level=self.logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        self.logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.logging.info(str(self.args))

    
    # -- rmtree : 디렉토리 및 파일 모두 삭제 후 파일 읽음
    def clean_old_ckpt(self):
        if os.path.exists('checkpoint/{}'.format(self.args.exp_name)):
            shutil.rmtree('checkpoint/{}'.format(self.args.exp_name))
            print(f'Clean the old checkpoint {self.args.exp_name}')
        os.mkdir('checkpoint/{}'.format(self.args.exp_name))
    
    # -- Tensorboard 시각화를 위한 로그 데이터 항목
    def make_writer(self):
        self.writer = SummaryWriter('checkpoint/{}'.format(self.args.exp_name)) 
    
    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list): # list형인지 확인
            dataset_list = []
            # train 리스트 안의 train1, train2를 하나의 데이터 리스트로 만듬
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    #add "img_HR, img_lr, label_str"
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len)) 
            train_dataset = dataset.ConcatDataset(dataset_list)
            print('??:',train_dataset)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=True) # imgH 32, imgW 128
        return train_dataset, train_loader

    # --- valid data --- #
    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    # --- test_data --- #
    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        test_dataset = self.load_dataset(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return test_dataset, test_loader

    # -- model 
    def generator_init(self):
        cfg = self.config.TRAIN
        
        # model 
        denoise_fn = Unet(dim=128, out_dim=3, dim_mults=(1, 2, 4), cond_dim=32).to(self.device)
        denoise_fn = torch.nn.DataParallel(denoise_fn)
        rrdb = RRDBNet(3, 3, 32, 8, 32//2)
        model = GaussianDiffusion(denoise_fn=denoise_fn, rrdb_net=rrdb, timesteps=100, loss_type='l2')
        image_crit = TPLoss(self.args)
        
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        image_crit.to(self.device)
        
        self.logging.info('loading pre-trained model from %s ' % self.resume)
        if self.resume is not '':
            self.logging.info('loading pre-trained model from %s ' % self.resume)
            model.load_state_dict(torch.load(self.resume)['state_dict_G'])
            # if self.config.TRAIN.ngpu == 1:
            #     model.load_state_dict(torch.load(self.resume)['state_dict_G'])
            # else:
            #     model.load_state_dict(
            #         {'module.' + k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})
        para_num = get_parameter_number(model)
        self.logging.info('Total Parameters {}'.format(para_num))
        
        return {'model': model, 'crit': image_crit}

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 200000, gamma=0.5)
    
    def build_optimizer(self, model):
        cfg = self.config.TRAIN
        params = list(model.named_parameters())
        if not True:
            params = [p for p in params if 'rrdb' not in p[0]]
        params = [p[1] for p in params]
        optimizer = optim.Adam(params, lr=cfg.lr,  betas=(cfg.beta1, 0.999))
        return optimizer
    
    def training_step(self, img_hr, img_lr, img_lr_up):#, batch):
        losses, _, _ = self.model(img_hr, img_lr, img_lr_up)
        total_loss = sum(losses.values())
        return losses, total_loss
    
    
    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(self.config.TRAIN.VAL.n_vis)):
            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            tensor_target = image_target[i][:3,:,:]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join('./demo', self.vis_dir)
            if not os.path.exists(out_root):
                os.mkdir(out_root)
            out_path = os.path.join(out_root, str(index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            if index is not 0:
                torchvision.utils.save_image(vis_im, os.path.join(out_path, im_name), padding=0)

    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, str_filt):
        visualized = 0
        for i in (range(image_in.shape[0])):
            if True:
                if (str_filt(pred_str_lr[i], 'lower') != str_filt(label_strs[i], 'lower')) and \
                        (str_filt(pred_str_sr[i], 'lower') == str_filt(label_strs[i], 'lower')):
                    visualized += 1
                    tensor_in = image_in[i].cpu()
                    tensor_out = image_out[i].cpu()
                    tensor_target = image_target[i].cpu()
                    transform = transforms.Compose(
                        [transforms.ToPILImage(),
                         transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                         transforms.ToTensor()]
                    )
                    tensor_in = transform(tensor_in)
                    images = ([tensor_in, tensor_out, tensor_target])
                    vis_im = torch.stack(images)
                    vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
                    out_root = os.path.join('./display', self.vis_dir)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
                    im_name = im_name.replace('/', '')
                    torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized
    
    def save_checkpoint(self, netG, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, exp_name):
        # ckpt_path = os.path.join('checkpoint', exp_name, self.vis_dir)
        ckpt_path = os.path.join('checkpoint', exp_name)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        save_dict = {
            'state_dict_G': netG.state_dict(),
            'info': {'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
            'best_history_res': best_acc_dict,
            'best_model_info': best_model_info,
            'param_num': sum([param.nelement() for param in netG.parameters()]),
            'converge': converge_list
        }
        if is_best:
            torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
        else:
            torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        self.logging.info('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        # MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN
    
    
    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = self.config.TRAIN.VAL.crnn_pretrained
        self.logging.info('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model, aster_info

    def parse_crnn_data(self, imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        self.logging.info('load pred_trained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict

    def CDistNet_init(self):
        dim_feedforward = 2048
        CDistNet_model = cdist.CDistNet(dim_feedforward=dim_feedforward)

        CDistNet_model = nn.DataParallel(CDistNet_model)
        CDistNet_model = CDistNet_model.to(self.device)

        model_path = self.config.TRAIN.VAL.cidist_pretrained
        self.logging.info('loading pre-trained CDistNet model from %s' % model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        
        CDistNet_model.load_state_dict(state_dict, strict=False)
        return CDistNet_model

    def parse_cdist_data(self, imgs_input, label_strs):
        batch_size = imgs_input.shape[0]
        tensor = torch.nn.functional.interpolate(imgs_input, size=(32, 128), mode='bicubic')
        all_chars = set(''.join(label_strs)) 
        vocab_size = 32 
        char_to_index = {char: idx for idx, char in enumerate(all_chars)} 
        max_length = max(len(label) for label in label_strs)
        label_indices = torch.full((batch_size, max_length), fill_value=-1, dtype=torch.long)  
        lengths = torch.zeros(batch_size, dtype=torch.int)  
        for i, label in enumerate(label_strs):
            lengths[i] = len(label)
            label_indices[i, :len(label)] = torch.tensor([char_to_index[char] for char in label], dtype=torch.long)
        return tensor, vocab_size, label_indices, lengths

class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)


to_pil = transforms.ToPILImage()

times = 0
easy_test_times = 0
medium_test_times = 0
hard_test_times = 0

class TextSR(TextBase):
    def train(self):
        cfg = self.config.TRAIN
        
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = 0    
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        scheduler.step(training_step)

        aster, aster_info = self.CRNN_init()
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()

        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []

        for epoch in range(cfg.epochs):
             for j, data in (enumerate(train_loader)):
                model.train()
                for p in model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j

                images_hr, images_lr, images_lr_up, label_strs = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                images_lr_up = images_lr_up.to(self.device)
                
                sr_img, rrdb_out = model.module.sample(images_lr, images_lr_up, images_hr.shape)
              
                # loss, mse_loss, attention_loss, recognition_loss = image_crit(sr_img, images_hr, label_strs)
                recognition_loss = image_crit(sr_img, images_hr, label_strs)
                ret, _, _ = model.module.forward(images_hr, images_lr, images_lr_up)
                mse_loss = ret['q']
                # loss = mse_loss + recognition_loss * 0.0005   #+ attention_loss * 10 
                loss = mse_loss + recognition_loss * 0.0005
                

                global times
                self.writer.add_scalar('loss/mse_loss', mse_loss , times)
                self.writer.add_scalar('loss/content_loss', recognition_loss, times)

                loss_im = loss * 100

                optimizer.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

                if iters % cfg.displayInterval == 0:
                    logging.info('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          # 'vis_dir={:s}\t'
                          'total_loss {:.3f} \t'
                          'mse_loss {:.3f} \t'
                          'recognition_loss {:.3f} \t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  # self.vis_dir,
                                  float(loss_im.data),
                                  mse_loss,
                                  recognition_loss
                                  ))
                
                if iters % cfg.VAL.valInterval == 0:
                    logging.info('======================================================')
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        logging.info('evaling %s' % data_name)
                        metrics_dict = self.eval(model, val_loader, image_crit, iters, aster, aster_info, data_name)
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:

                            data_for_evaluation = metrics_dict['images_and_labels']

                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                
                if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        logging.info('saving best model')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True,
                                             converge_list, self.args.exp_name)
                        
                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list,
                                         self.args.exp_name)

    def get_crnn_pred(self, outputs):
        alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
        predict_result = []
        for output in outputs:
            max_index = torch.max(output, 1)[1]
            out_str = ""
            last = ""
            for i in max_index:
                if alphabet[i] != last:
                    if i != 0:
                        out_str += alphabet[i]
                        last = alphabet[i]
                    else:
                        last = ""
            predict_result.append(out_str)
        return predict_result

    

    def eval(self, model, val_loader, image_crit, index, recognizer, aster_info, mode):
        global easy_test_times    
        global medium_test_times    
        global hard_test_times   

        for p in model.parameters():
            p.requires_grad = False
        for p in recognizer.parameters():
            p.requires_grad = False
        model.eval()
        recognizer.eval()
        n_correct = 0
        n_correct_lr = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,
                       'images_and_labels': []}
        image_start_index = 0

        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, data in pbar:
            # if self.first_val and batch_idx > 

            images_hr, images_lr, images_lr_up, label_strs = data
            
            
            val_batch_size = images_lr.shape[0]
            images_hr = images_hr.to(self.device)
            images_lr = images_lr.to(self.device)
            images_lr_up = images_lr_up.to(self.device)

            # inference time
            start_time = time.time()

            images_sr, rrdb_out = model.module.sample(images_lr, images_lr_up, images_hr.shape)
            
            # End inference time
            inference_time_model = time.time() - start_time

            if i == len(val_loader) - 1:
                index = random.randint(0, images_lr.shape[0]-1)
                self.writer.add_image(f'vis/{mode}/lr_image', images_lr[index,...], easy_test_times)
                self.writer.add_image(f'vis/{mode}/sr_image', images_sr[index,...], easy_test_times)
                self.writer.add_image(f'vis/{mode}/hr_image', images_hr[index,...], easy_test_times)
            
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_psnr(images_sr, images_hr))

            recognizer_dict_sr = self.parse_crnn_data(images_sr[:, :3, :, :])

            # recognizer inference
            start_time = time.time()

            recognizer_output_sr = recognizer(recognizer_dict_sr)
            
            # End Recognizer inference
            inference_time_recognizer = time.time() - start_time

            outputs_sr = recognizer_output_sr.permute(1, 0, 2).contiguous()
            predict_result_sr = self.get_crnn_pred(outputs_sr)
            metric_dict['images_and_labels'].append(
                (images_hr.detach().cpu(), images_sr.detach().cpu(), label_strs, predict_result_sr))
            
            

            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        logging.info('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      0, 0,
                      float(psnr_avg), float(ssim_avg), ))
        logging.info('save display images')
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        logging.info('sr_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        if mode == 'easy':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, easy_test_times)
            easy_test_times += 1
        if mode == 'medium':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, medium_test_times)
            medium_test_times += 1
        if mode == 'hard':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, medium_test_times)
            
            hard_test_times += 1

        # print(f"Model Inference Time: {inference_time_model} seconds")
        # print(f"Recognizer Inference Time: {inference_time_recognizer} seconds")
        # print(f"easy time : {easy_test_time}")
        # print(f"easy time : {medium_test_time}")
        # print(f"easy time : {hard_test_time}")
        return metric_dict
    
    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        items = os.listdir(self.test_data_dir)
        for test_dir in items:
            test_data, test_loader = self.get_test_data(os.path.join(self.test_data_dir, test_dir))
            data_name = self.args.test_data_dir.split('/')[-1]
            logging.info('evaling %s' % data_name)
            if self.args.rec == 'moran':
                moran = self.MORAN_init()
                moran.eval()
            elif self.args.rec == 'aster':
                aster, aster_info = self.Aster_init()
                aster.eval()
            elif self.args.rec == 'crnn':
                crnn, _ = self.CRNN_init()
                crnn.eval()
            elif self.args.rec == 'cdist':
                cdist = self.CDistNet_init()
                cdist.eval()
            # if self.args.arch != 'bicubic':
            #     for p in model.parameters():
            #         p.requires_grad = False
                model.eval()
            n_correct = 0
            sum_images = 0
            metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
            current_acc_dict = {test_dir: 0}
            time_begin = time.time()
            sr_time = 0
            for i, data in (enumerate(test_loader)):
                images_hr, images_lr, images_lr_up, label_strs = data
                val_batch_size = images_lr.shape[0]
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                images_lr_up = images_lr_up.to(self.device)
                sr_beigin = time.time()
                images_sr, rrdb_out = model.module.sample(images_lr, images_lr_up, images_hr.shape)
                sr_end = time.time()
                sr_time += sr_end - sr_beigin
                metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                if self.args.rec == 'moran':
                    moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                    moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                         debug=True)
                    preds, preds_reverse = moran_output[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                    pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
                elif self.args.rec == 'aster':
                    aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                    aster_output_sr = aster(aster_dict_sr["images"])
                    pred_rec_sr = aster_output_sr['output']['pred_rec']
                    pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                    aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                    aster_output_lr = aster(aster_dict_lr)
                    pred_rec_lr = aster_output_lr['output']['pred_rec']
                    pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                elif self.args.rec == 'crnn':
                    crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                    crnn_output = crnn(crnn_input)
                    _, preds = crnn_output.max(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                    pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                elif self.args.rec == 'cdist':
                    tensor, vocab_size, label_indices, lengths = self.parse_cdist_data(images_sr[:, :3, :, :], label_strs)
                    tgt = F.pad(label_indices, (1, 0), "constant", 0)

                    def validate_and_correct_indices(tgt, vocab_size):
                        corrected_tgt = tgt.clone()  
                        out_of_range_indices = (corrected_tgt >= vocab_size) | (corrected_tgt < 0)
                        corrected_tgt[out_of_range_indices] = vocab_size - 1  
                        return corrected_tgt
                    
                    corrected_tgt = validate_and_correct_indices(tgt, vocab_size)
                    tgt = corrected_tgt.to(self.device)
                    cdist_output = cdist(tensor, tgt)
                    #print('cdist_output:', cdist_output)

                    alphabet = 'abcdefghijklmnopqrstuvwxyz'
                    converter = utils_cdist.SimpleConverterForCDistNet(alphabet)
                    pred_str_sr = converter.decode(cdist_output)

                for pred, target in zip(pred_str_sr, label_strs):
                    if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                        n_correct += 1

                sum_images += val_batch_size
                torch.cuda.empty_cache()
                # if i % 10 == 0:
                #     logging.info('Evaluation: [{}][{}/{}]\t'
                #           .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                #                   i + 1, len(test_loader), ))
                # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
            time_end = time.time()
            total_time = time_end - time_begin

            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
            acc = round(n_correct / sum_images, 4)
            fps = sum_images / (time_end - time_begin)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            current_acc_dict[test_dir] = float(acc)
            result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
            
            logging.info(result)
            print(f'Sum Images : {sum_images}')
            print(f"Total Inference Time: {total_time} seconds\n")
            print(f"SR Model Inference Time: {sr_time} seconds\n")
            