import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

class LCATSSR(nn.Module):
    def __init__(self, up_scale=4, dim=64, groups=6,num=5):
        super(LCATSSR, self).__init__()
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
    def forward(self, fea):
        x_left0,x_right0  = fea.chunk(2, dim=1)
        x_left = self.init(x_left0)
        x_right = self.init(x_right0)
        for i in range(self.groups):
            x_left,x_right  = self.body[i](x_left,x_right)
        x_left = self.up(x_left) + F.interpolate(x_left0, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        x_right = self.up(x_right) + F.interpolate(x_right0, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        return torch.cat([x_left,x_right],dim=1)

class GroupSR(nn.Module):
    def __init__(self, dim, num=6,wsize=16):
        super().__init__()
        self.num = num
        self.body = nn.ModuleList()
        for i in range(num):
            self.body.append(BasicBlockSR(dim,interaction=(i==0),spatial=(i!=0 and i%2==1),channel=(i!=0 and i%2==0),window_sizes = wsize ,shift=((i+1)%4==0)))
        self.conv = nn.Conv2d(dim,dim,1)
    def forward(self,x_left0,x_right0):
        x_left,x_right=x_left0,x_right0
        for i in range(self.num):
            x_left,x_right = self.body[i](x_left,x_right)
        x_left = self.conv(x_left)
        x_right = self.conv(x_right)
        return x_left+x_left0,x_right+x_right0

class BasicBlockSR(nn.Module):
    def __init__(self, dim, interaction = False, spatial = False, channel = False, window_sizes = 8,shift = False):
        super().__init__()
        self.interaction = Crossattention(dim,a=0.75) if interaction else None
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
    def forward(self, x_left,x_right):
        b,c,h,w=x_left.shape
        if self.interaction:
            x_left,x_right = self.interaction(x_left,x_right)
        if self.spatial:
            x_left = self.check_image_size2(x_left,self.window_sizes)
            x_right = self.check_image_size2(x_right,self.window_sizes)
            x_left = self.spatial(x_left)[:,:,:h,:w]
            x_right = self.spatial(x_right)[:,:,:h,:w]
        if self.channel:
            x_left = self.channel(x_left)
            x_right = self.channel(x_right)
        x_left = self.MLP(x_left)
        x_right = self.MLP(x_right)
        return x_left,x_right

class Crossattention(nn.Module):
    def __init__(self,dim,a=0.5):
        super(Crossattention,self).__init__()
        adim = int(dim*a)
        self.LayernormL = LayerNorm2d(dim)
        self.LayernormR = LayerNorm2d(dim)
        self.feaL = nn.Conv2d(dim,dim,1,bias=False)
        self.feaR = nn.Conv2d(dim,dim,1,bias=False)
        self.to_l = nn.Conv2d(dim,adim,1,bias=False)
        self.to_r = nn.Conv2d(dim,adim,1,bias=False)
        self.transition = nn.Sequential(nn.Conv2d(dim,dim,3,1,1,groups=dim),nn.Conv2d(dim,dim,1))
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Conv2d(dim,dim,1,bias=False)
    def forward(self,xl0,xr0):
        B,C,H,W= xl0.shape
        vl = self.feaL(xl0)
        vr = self.feaR(xr0)
        xl = self.transition(xl0)+xl0
        xr = self.transition(xr0)+xr0
        xl = self.LayernormL(xl)
        xl = self.to_l(xl)
        xr = self.LayernormR(xr)
        xr = self.to_r(xr)
        xl = rearrange(xl,'b c h w-> (b h) w c')
        xr = rearrange(xr,'b c h w-> (b h) w c')
        vl = rearrange(vl,'b c h w-> (b h) w c')
        vr = rearrange(vr,'b c h w-> (b h) w c')
        atn = torch.bmm(xl,xr.permute(0,2,1))
        warpL = torch.bmm(atn.softmax(dim=-1),vr)
        warpR = torch.bmm(atn.permute(0,2,1).softmax(dim=-1),vl)
        warpL = rearrange(warpL,'(b h) w c->b c h w',h=H)
        warpR = rearrange(warpR,'(b h) w c->b c h w',h=H)
        xl = self.out(warpL)
        xr = self.out(warpR)
        return xl+xl0,xr+xr0
    
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
        self.Layernorm = LayerNorm2d(dim)
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
        self.Layernorm = LayerNorm2d(dim)
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
        self.layernorm1 = LayerNorm2d(dim)
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

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

if __name__ == "__main__":
    net = LCATSSR(up_scale=4,  dim=64, groups=6,num=5)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
