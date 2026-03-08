import math
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, einsum

class CAB(nn.Module):
    def __init__(self, poolin_channel, out_channel):
        super(CAB, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 最大赤化还没试过
        self.conv1 = nn.Conv2d(poolin_channel, out_channel, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, globle_pool):
        out1 = x
        out1_1 = torch.cat([x, globle_pool], 1)
        out1_2 = self.pool(out1_1)
        out2 = self.sigmoid(self.conv2(self.bn(self.relu(self.conv1(out1_2)))))
        out3 = out1 * out2
        out = torch.cat([out3, globle_pool], 1)
        # out = out3+globle_pool
        return out
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim_in, dim_out):
    return nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

def Downsample(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    
    def __init__(self, dim, dim_out, *, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) #if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) #chunk沿通道维度（dim=1）均匀分割为3部分
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class NetworkConfig:

    image_channels=3
    n_classes=19
    dim=32
    dim_mults=(1, 2, 4, 8)
    resnet_block_groups=8

    # diffusion parameters
    n_timesteps = 10
    n_scales = 3
    max_patch_size = 512

    # ensemble parameters
    built_in_ensemble = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Network(nn.Module):
    def __init__(
            self,
            network_config=NetworkConfig(),
            ): 
        super().__init__()
        self.config = network_config
        image_channels = self.config.image_channels
        n_classes = self.config.n_classes
        dim = self.config.dim
        dim_mults = self.config.dim_mults
        resnet_block_groups = self.config.resnet_block_groups

        # determine dimensions
        self.image_channels = image_channels
        self.n_classes = n_classes
        self.dims = [c * dim for c in dim_mults]

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # image initial 图像初始化处理模块
        self.image_initial = nn.ModuleList([
            ResNetBlock(image_channels, self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # segmentation initial 分割图初始化处理模块
        self.seg_initial = nn.ModuleList([
            ResNetBlock(n_classes, self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # layers
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.CAB = nn.ModuleList([])
        self.conv = nn.ModuleList([])
        # encoder
        for i in range(len(dim_mults)-1): # each dblock
            dim_in = self.dims[i]
            dim_out = self.dims[i+1]

            self.down.append(
                nn.ModuleList([
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out),
                    
                ])
            )
                
        # decoder
        for i in range(len(dim_mults)-1): # each ublock
            dim_in = self.dims[-i-1]
            dim_out = self.dims[-i-2]
            if i == 0:
                dim_in_plus_concat = dim_in
            else:
                dim_in_plus_concat = dim_in * 2
            
            self.up.append(
                nn.ModuleList([
                    ResNetBlock(dim_in_plus_concat, dim_in, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in, dim_out),
                ])
            )
        for i in range(len(dim_mults) - 1):
            self.CAB.append(CAB(poolin_channel=self.dims[-i - 1], out_channel=self.dims[-i - 2]))
            self.conv.append(nn.Conv2d(self.dims[-i - 2], self.dims[-i - 1], 1))

        self.final = nn.Sequential(ResNetBlock(self.dims[0]*2, self.dims[0], groups=resnet_block_groups), 
                                   ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
                                   nn.Conv2d(self.dims[0], n_classes, 1))

        self.dilated_convolutions = nn.ModuleList([
            nn.Conv2d(self.dims[3], self.dims[3], 3, padding=2, dilation=1),
            nn.Conv2d(self.dims[3], self.dims[3], 3, padding=4, dilation=6),
            nn.Conv2d(self.dims[3], self.dims[3], 3, padding=8, dilation=12),
            nn.Conv2d(self.dims[3], self.dims[3], 3, padding=16, dilation=18)
        ])
        self.Channel_Attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.dims[3], self.dims[3]//8, 1), nn.ReLU(), nn.Conv2d(self.dims[3]//8, self.dims[3], 1), nn.Sigmoid())
        self.Spatial_Attention = nn.Sequential(nn.Conv2d(self.dims[3], self.dims[3]//8, 1), nn.ReLU(), nn.Conv2d(self.dims[3]//8, self.dims[3], 1), nn.Sigmoid())
    def forward(self, seg, img):

        resnetblock1, resnetblock2, resnetblock3 = self.seg_initial
        seg_emb = resnetblock1(seg)
        seg_emb = resnetblock2(seg_emb)
        seg_emb = resnetblock3(seg_emb)

        resnetblock1, resnetblock2, resnetblock3 = self.image_initial
        img_emb = resnetblock1(img)
        img_emb = resnetblock2(img_emb)
        img_emb = resnetblock3(img_emb)

        x = seg_emb + img_emb

        h = []

        for resnetblock1, resnetblock2, attn, downsample in self.down:
            x = resnetblock1(x)
            x = resnetblock2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        i = 0
        for  resnetblock1, resnetblock2, attn, upsample in self.up:
            x = resnetblock1(x)
            x = resnetblock2(x)
            x = attn(x)
            x = upsample(x)
            x = self.CAB[i](x, h.pop())
            x = self.conv[i](x)
            x = torch.cat((x, h.pop()), dim=1)
            i=i+1
        return self.final(x)