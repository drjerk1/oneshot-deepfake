import torch
from torch import tensor
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch import autograd
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter
import math

def disable_spectral_norm(x):
    return x

class Attention(nn.Module):
    def __init__(self, channels, query_channels, value_channels, spectral_norm, heads=1):
        super().__init__()

        self.conv_query = spectral_norm(nn.Conv2d(channels, query_channels, kernel_size=1, bias=False))
        self.conv_key = spectral_norm(nn.Conv2d(channels, query_channels, kernel_size=1, bias=False))
        self.conv_value = spectral_norm(nn.Conv2d(channels, value_channels, kernel_size=1, bias=False))
        self.conv_sc = spectral_norm(nn.Conv2d(channels, value_channels, kernel_size=1, bias=False))
        self.heads = heads
        self.div_ch = query_channels // heads
        self.div_ch_v = value_channels // heads
        self.gamma = nn.Parameter(torch.zeros(1, value_channels, 1, 1), requires_grad=True)
        assert query_channels % heads == 0
        assert value_channels % heads == 0

        nn.init.orthogonal_(self.conv_query.weight.data)
        nn.init.orthogonal_(self.conv_key.weight.data)
        nn.init.orthogonal_(self.conv_value.weight.data)
        nn.init.orthogonal_(self.conv_sc.weight.data)

    def forward(self, x):
        b, c, w, h = x.shape

        proj_query = self.conv_query(x).view(b * self.heads, self.div_ch, w, h).permute(0, 2, 3, 1).contiguous()
        # b H w h cH
        proj_query = proj_query.view(b * self.heads, w * h, self.div_ch)
        # b H w*h cH
        proj_key = self.conv_key(x).view(b * self.heads, self.div_ch, w * h)
        # b H cH w*h

        attn = torch.bmm(proj_query, proj_key)
        # O(b * w^2 * h^2 * c)
        # (b H w*h cH) * (b H cH w*h) -> (b H w*h w*h)
        attn = F.softmax(attn, dim=-1)
        # b H w*h (w*h probs)

        proj_value = self.conv_value(x).view(b * self.heads, self.div_ch_v, w, h).permute(0, 2, 3, 1).contiguous()
        # b H w h cH
        proj_value = proj_value.view(b * self.heads, w*h, self.div_ch_v)
        # b H w*h cH
        attn = torch.bmm(attn, proj_value)
        # O(b * w^2 * h^2 * c)
        # (b H w*h w*h) * (b H w*h cH) -> (b, H, w*h, cH)
        attn = attn.view(b, self.heads, w*h, self.div_ch_v)
        attn = attn.permute(0, 2, 1, 3).contiguous()
        # b w*h H CH
        attn = attn.view(b, w, h, self.div_ch_v * self.heads)
        # b w h c
        attn = attn.permute(0, 3, 1, 2).contiguous()
        # b c w h
        return self.gamma * attn + self.conv_sc(x)

class CBN2d(nn.Module):
    def __init__(self, num_features, num_conditions, spectral_norm):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = spectral_norm(nn.Conv2d(num_conditions, num_features*2, kernel_size=1, bias=False))

        nn.init.orthogonal_(self.embed.weight.data)

    def forward(self, x, y):
        out = self.bn(x)
        embed = self.embed(y.unsqueeze(2).unsqueeze(3))
        gamma, beta = embed.chunk(2, dim=1)
        out = (1.0 + gamma.contiguous()) * out + beta.contiguous()

        return out

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, optimized, spectral_norm, use_bn=True):
        super().__init__()
        self.downsample = downsample
        self.optimized = optimized
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.learnable_sc = in_channels != out_channels or downsample

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.relu = nn.ReLU()

        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        if self.learnable_sc:
            nn.init.orthogonal_(self.conv_sc.weight.data)

    def _residual(self, x):
        if not self.optimized:
            x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.use_bn:
            x = self.bn(x)
        return x

    def _shortcut(self, x):
        if self.learnable_sc:
            if self.optimized:
                x = self.conv_sc(F.avg_pool2d(x, 2)) if self.downsample else self.conv_sc(x)
            else:
                x = F.avg_pool2d(self.conv_sc(x), 2) if self.downsample else self.conv_sc(x)

        return x

    def forward(self, x):
        return self._shortcut(x) + self._residual(x)

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conditions, upsample, spectral_norm, use_bn=True):
        super().__init__()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.cbn1 = CBN2d(in_channels, num_conditions, spectral_norm=spectral_norm)
        self.cbn2 = CBN2d(out_channels, num_conditions, spectral_norm=spectral_norm)
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.relu = nn.ReLU()

        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        if self.learnable_sc:
            nn.init.orthogonal_(self.conv_sc.weight.data)

    def _upsample_conv(self, x, conv):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = conv(x)

        return x

    def _residual(self, x, y):
        x = self.relu(self.cbn1(x, y))
        x = self._upsample_conv(x, self.conv1) if self.upsample else self.conv1(x)
        x = self.relu(self.cbn2(x, y))
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn(x)
        return x

    def _shortcut(self, x):
        if self.learnable_sc:
            x = self._upsample_conv(x, self.conv_sc) if self.upsample else self.conv_sc(x)

        return x

    def forward(self, x, y):
        return self._shortcut(x) + self._residual(x, y)

class BigEncoder(nn.Module):
    def __init__(self, num_images, image_size, ch, use_attn, attn_layer, spectral_norm, deep):
        super().__init__()

        self.attn = None
        if use_attn:
            self.attn = []
        self.attn_layer = attn_layer
        self.image_size = image_size
        self.num_images = num_images

        blocks = []
        log_im_size = int(math.log(image_size, 2))
        assert math.log(image_size, 2) == log_im_size and image_size >= 32
        w = self.image_size
        first_block = True
        from_ch = num_images
        assert ch % (2 ** (log_im_size - 4)) == 0
        to_ch = ch // (2 ** (log_im_size - 4))

        while w > 4:
            blocks.append(DBlock(from_ch,
                                 to_ch,
                                 downsample=True,
                                 optimized=first_block,
                                 spectral_norm=spectral_norm))
            if deep:
                blocks.append(DBlock(to_ch,
                                     to_ch,
                                     downsample=False,
                                     optimized=False,
                                     spectral_norm=spectral_norm))
            w //= 2
            if use_attn and w in self.attn_layer:
                self.attn.append(Attention(to_ch, to_ch, to_ch, spectral_norm=spectral_norm))
            first_block = False
            from_ch = to_ch
            if w > 8:
                to_ch *= 2

        self.blocks = nn.ModuleList(blocks)
        if self.attn is not None:
            self.attn = nn.ModuleList(self.attn)

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[1] == self.num_images and x.shape[2] == x.shape[3] and x.shape[3] == self.image_size

        skip = []

        w = self.image_size
        k = 0
        for block in self.blocks:
            x = block(x)
            if block.downsample:
                w //= 2
                if (self.attn is not None) and w in self.attn_layer:
                    x = self.attn[k](x)
                    k += 1
            skip.append(x)

        return skip

class BigGenerator(nn.Module):
    def __init__(self, num_images,
                 image_size,
                 latent_dim_mul=20,
                 ch=512,
                 num_classes=0,
                 sparse_num_classes=False,
                 encoder_image_size=0,
                 encoder_num_images=0,
                 use_attn=True,
                 attn_layer=[4, 8, 16, 32],
                 spectral_norm=disable_spectral_norm,
                 cond_spectral_norm=disable_spectral_norm,
                 encoder_spectral_norm=disable_spectral_norm,
                 encoder_use_attn=True,
                 encoder_attn_layer=[4, 8, 16, 32],
                 deep=False):
        super().__init__()

        log_im_size = int(math.log(image_size, 2))
        assert math.log(image_size, 2) == log_im_size and image_size >= 32

        self.ch = ch
        self.num_images = num_images
        self.encoder_num_images = encoder_num_images
        self.sparse_num_classes = sparse_num_classes
        self.image_size = image_size
        self.encoder_image_size = encoder_image_size
        self.latent_dim = latent_dim_mul * (log_im_size - 1)
        self.num_classes = num_classes
        self.attn = None
        if use_attn:
            self.attn = []
        self.attn_layer = attn_layer
        self.num_chunk = log_im_size - 1
        if deep:
            self.num_chunk *= 2

        if self.num_classes > 0:
            layer = nn.Embedding if sparse_num_classes else nn.Linear
            self.embed = cond_spectral_norm(layer(self.num_classes, self.num_classes))
            nn.init.orthogonal_(self.embed.weight.data)

        if self.encoder_num_images > 0:
            self.encoder = BigEncoder(num_images=self.encoder_num_images,
                                      image_size=self.encoder_image_size,
                                      ch=ch,
                                      use_attn=encoder_use_attn,
                                      attn_layer=encoder_attn_layer,
                                      spectral_norm=encoder_spectral_norm,
                                      deep=deep)

        num_latents = self.__get_num_latents()

        blocks = []
        w = 4
        from_ch = ch
        to_ch = from_ch

        self.fc = spectral_norm(nn.Linear(num_latents[0], from_ch*4*4, bias=False))
        idx = 1
        while w < self.image_size:
            assert to_ch % 2 == 0
            if deep:
                blocks.append(GBlock(from_ch, from_ch,
                                     num_latents[idx],
                                     upsample=False,
                                     spectral_norm=spectral_norm))
                idx += 1

            if w == 4 and use_attn and w in self.attn_layer:
                self.attn.append(Attention(from_ch, from_ch, from_ch, spectral_norm=spectral_norm))

            blocks.append(GBlock(from_ch, to_ch, num_latents[idx],
                                 upsample=True,
                                 spectral_norm=spectral_norm))
            w *= 2

            if use_attn and w in self.attn_layer:
                self.attn.append(Attention(to_ch, to_ch, to_ch, spectral_norm=spectral_norm))

            from_ch = to_ch
            to_ch //= 2
            idx += 1

        assert from_ch >= self.num_images * 4

        self.blocks = nn.ModuleList(blocks)
        if self.attn is not None:
            self.attn = nn.ModuleList(self.attn)

        self.bn = nn.BatchNorm2d(from_ch)
        self.relu = nn.ReLU()
        self.conv_last = spectral_norm(nn.Conv2d(from_ch, self.num_images, kernel_size=3, padding=1, bias=False))
        self.tanh = nn.Tanh()

        nn.init.orthogonal_(self.fc.weight.data)
        nn.init.orthogonal_(self.conv_last.weight.data)
        nn.init.constant_(self.bn.weight.data, 1.0)
        nn.init.constant_(self.bn.bias.data, 0.0)

    def __get_num_latents(self):
        xs = torch.empty(self.latent_dim).chunk(self.num_chunk)
        num_latents = [x.size(0) for x in xs]
        for i in range(1, self.num_chunk):
            num_latents[i] += self.num_classes

        return num_latents

    def forward(self, x, y=None, z=None):
        assert len(x.shape) == 2 and x.shape[-1] == self.latent_dim

        if self.num_classes > 0:
            assert y is not None
            assert y.shape[0] == x.shape[0]
            if not self.sparse_num_classes:
                assert len(y.shape) == 2
                assert y.shape[1] == self.num_classes
            else:
                assert len(y.shape) == 1
            y = self.embed(y)

        if self.encoder_num_images > 0:
            assert z is not None
            assert x.shape[0] == z.shape[0]
            encoded = self.encoder(z)
        else:
            encoded = []

        xs = x.chunk(self.num_chunk, dim=1)

        h = self.fc(xs[0])
        h = h.view(h.shape[0], self.ch, 4, 4)

        i = 0
        j = len(encoded) - 1
        k = 0

        w = 4
        while i < len(self.blocks):
            if j >= 0:
                h = h * encoded[j]
                j -= 1
            if self.num_classes > 0:
                cond = torch.cat([y, xs[i + 1]], dim=1)
            else:
                cond = xs[i + 1]

            if i == 0 and w == 4 and (self.attn is not None) and w in self.attn_layer and not(self.blocks[i].upsample):
                h = self.blocks[i](h, cond)
                h = self.attn[k](h)
                k += 1
            elif i == 0 and w == 4 and (self.attn is not None) and w in self.attn_layer and self.blocks[i].upsample:
                h = self.attn[k](h)
                k += 1
                h = self.blocks[i](h, cond)
            else:
                h = self.blocks[i](h, cond)

            if self.blocks[i].upsample:
                w *= 2
                if (self.attn is not None) and w in self.attn_layer:
                    h = self.attn[k](h)
                    k += 1
            i += 1

        assert i == len(self.blocks) and j == max(len(encoded) - len(self.blocks) - 1, -1)

        return self.tanh(self.conv_last(self.relu(self.bn(h))))
