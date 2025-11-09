import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

class LCATSR(nn.Module):
    def __init__(self, up_scale=4, dim=60, groups=5,num=4):
        super(LCATSR, self).__init__()
        self.init = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        self.num = num
        self.body = nn.ModuleList()
        self.groups = groups
        for i in range(groups):
            self.body.append(GroupSR(dim,num=num,wsize=16))
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3 * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
    def forward(self, x0):
        x = self.init(x0)
        for i in range(self.groups):
            x  = self.body[i](x)
        x = self.up(x) + F.interpolate(x0, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        return x
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('up') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('up') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class GroupSR(nn.Module):
    def __init__(self, dim, num=6,wsize=16):
        super().__init__()
        self.num = num
        self.body = nn.ModuleList()
        for i in range(num):
            self.body.append(BasicBlockSR(dim,spatial=(i%2==0),channel=(i%2==1),window_sizes = wsize ,shift=((i+2)%4==0)))
        self.conv = nn.Conv2d(dim,dim,1)
    def forward(self,x0):
        x=x0
        for i in range(self.num):
            x = self.body[i](x)
        x = self.conv(x)
        return x+x0

class BasicBlockSR(nn.Module):
    def __init__(self, dim, spatial = False, channel = False, window_sizes = 8,shift = False):
        super().__init__()
        self.spatial = SelfAttention(dim,heads=2,a=0.5,wsize=window_sizes,shift=shift) if spatial else None
        self.channel = ChannelAttention(dim,a=0.5,heads=1) if channel else None
        self.window_sizes = window_sizes
        self.MLP = MLP(dim,ratio=2)
    def check_image_size2(self, x, wsize):
        _, _, h, w = x.size()
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, x):
        b,c,h,w=x.shape
        if self.spatial:
            x = self.check_image_size2(x,self.window_sizes)
            x = self.spatial(x)[:,:,:h,:w]
        if self.channel:
            x = self.channel(x)
        x = self.MLP(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self,dim,heads=1,a=0.5,wsize = 8,shift =False):
        super().__init__()
        adim = int(dim * a)
        bdim = dim-adim
        self.dims = [adim,bdim]
        self.qk = nn.Conv2d(dim,adim*2,1)
        self.vs = nn.Conv2d(dim,dim,1)
        self.wsize = wsize
        self.scale = (adim//heads) ** -0.5
        self.shift = shift
        self.softmax = nn.Softmax(dim=-1)
        self.Layernorm = LayerNorm(dim)
        self.depth = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.fusion = Fusion(dim,4)
        self.proj = nn.Conv2d(dim,dim,1)
        self.heads = heads
    def forward(self,x0):
        b,c,h,w = x0.shape
        x = self.Layernorm(x0)
        q,k = self.qk(x).chunk(2,dim=1)
        vs = self.vs(x)
        local = self.depth(vs)
        v1,v2 = vs.split(self.dims,dim=1)
        if self.shift:
            q = torch.roll(q,shifts=(-self.wsize//2, -self.wsize//2), dims=(2,3))
            k = torch.roll(k,shifts=(-self.wsize//2, -self.wsize//2), dims=(2,3))
            v1 = torch.roll(v1,shifts=(-self.wsize//2, -self.wsize//2), dims=(2,3))
        q = rearrange(q,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        k = rearrange(k,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        v = rearrange(v1,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        atn = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        atn = self.softmax(atn)
        y = torch.matmul(atn,v)
        y = rearrange(y,'(b h w) hed (dh dw) c->b (hed c) (h dh) (w dw)',h = h//self.wsize,w=w//self.wsize,dh=self.wsize,dw=self.wsize)
        if self.shift:
            y =  torch.roll(y, shifts=(self.wsize//2, self.wsize//2), dims=(2, 3))
        y = torch.cat([y,v2],dim=1)
        y = self.fusion(y,local)
        y = self.proj(y)
        return y+x0

class ChannelAttention(nn.Module):
    def __init__(self,dim,a=0.5,heads=1):
        super().__init__()
        adim = int(dim * a)
        bdim = dim-adim
        self.dims = [adim,bdim]
        self.Layernorm = LayerNorm(dim)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qk = nn.Sequential(nn.Conv2d(dim,adim*2,1),nn.Conv2d(adim*2,adim*2,3,1,1,groups=adim*2))
        self.v = nn.Sequential(nn.Conv2d(dim,dim,1),nn.Conv2d(dim,dim,3,1,1,groups=dim))
        self.project_out = nn.Conv2d(dim, dim, 1)
        self.num_heads = heads
        self.depth = nn.Sequential(nn.Conv2d(dim, dim//4, 1,bias=False), nn.LeakyReLU(0.2),nn.Conv2d(dim//4, dim, 1,bias=False),nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = Fusion(dim,4)
    def forward(self,x0):
        b,c,h,w = x0.shape
        x = self.Layernorm(x0)
        q,k = self.qk(x).chunk(2,dim=1)
        vs = self.v(x)
        local = self.depth(self.avg_pool(vs)) * vs
        v1,v2 = vs.split(self.dims,dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.matmul(q,k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        y = torch.matmul(attn,v)
        y = rearrange(y, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        y = torch.cat([y,v2],dim=1)
        y =  self.fusion(y,local)
        y = self.project_out(y)
        return y + x0

class Fusion(nn.Module):
    def __init__(self, in_channels,reduction=4):
        super(Fusion, self).__init__()
        redim = in_channels//reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, redim, 1,bias=False), nn.LeakyReLU(0.2),nn.Conv2d(redim, in_channels, 1,bias=False),nn.Sigmoid())
    def forward(self, x1,x2):
        x = x1+x2
        x = self.avg_pool(x)
        ca = self.conv_du(x)
        x = x1*ca +x2*(1-ca)
        return x
            
class MLP(nn.Module):
    def __init__(self, dim,ratio=2):
        super(MLP,self).__init__()
        self.layernorm1 = LayerNorm(dim)
        expandim = int(dim*ratio)
        self.proj1 = nn.Conv2d(dim,expandim,1)
        self.conv = nn.Conv2d(expandim,expandim,3,1,1,groups=expandim)
        self.projout = nn.Conv2d(dim,dim,1)
    def forward(self,x0):
        x = self.layernorm1(x0)
        x = self.proj1(x)
        x1,x2 = self.conv(x).chunk(2,dim=1)
        x = F.gelu(x1) * x2
        x = self.projout(x)
        return x + x0

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight

# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        return to_4d(x, h, w)

if __name__ == "__main__":
    net = LCATSR(up_scale=2,  dim=60, groups=5,num=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fK' % (total / 1e3))
