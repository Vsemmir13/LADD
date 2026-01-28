import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from edm.torch_utils import persistence, misc
from edm.training.networks import Linear, PositionalEmbedding, silu
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Conditional Unet defined below
class CondINorm(nn.Module):
    def __init__(self, in_channels, noise_channels, eps=1e-5):
        super(CondINorm, self).__init__()
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv2d(noise_channels, in_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(noise_channels, in_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x, z):
        shift = self.shift_conv.forward(z)
        scale = self.scale_conv.forward(z)
        size = x.size()
        x_reshaped = x.view(size[0], size[1], size[2] * size[3])
        mean = x_reshaped.mean(2, keepdim=True)
        var = x_reshaped.var(2, keepdim=True)
        std = torch.rsqrt(var + self.eps)
        norm_features = ((x_reshaped - mean) * std).view(*size)
        output = norm_features * scale + shift
        return output


class CondDoubleConv(nn.Module):
    """(convolution => [CIN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, noise_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = CondINorm(mid_channels, noise_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = CondINorm(out_channels, noise_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, z):
        x = self.conv1(x)
        x = self.norm1(x, z)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, z)
        x = self.relu2(x)
        return x


class CondUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, noise_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = CondDoubleConv(
            in_channels, out_channels, noise_channels, in_channels // 2
        )

    def forward(self, x1, x2, z):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, z)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class CUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        label_dim=0,
        noise_channels=128,
        base_factor=128,
        emb_channels=128,
        embedding_type="positional",
        **kwargs,
    ):
        super(CUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.noise_channels = noise_channels
        self.base_factor = base_factor

        self.inc = DoubleConv(in_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor)
        factor = 2
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        self.adain1 = CondINorm(16 * base_factor // factor, noise_channels)
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor)
        self.adain2 = CondINorm(8 * base_factor // factor, noise_channels)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor)
        self.adain3 = CondINorm(4 * base_factor // factor, noise_channels)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor)
        self.adain4 = CondINorm(2 * base_factor // factor, noise_channels)
        self.up4 = Up(2 * base_factor, base_factor)
        self.outc = OutConv(base_factor, out_channels)

        init = dict(init_mode="xavier_uniform")
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = (
            Linear(in_features=label_dim, out_features=noise_channels, **init)
            if label_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )

    def forward(self, x, noise_labels, class_labels, return_features=False):
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))

        emb = silu(self.map_layer0(emb))
        z = silu(self.map_layer1(emb)).unsqueeze(-1).unsqueeze(-1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        features = [x1, x2, x3, x4, x5] if return_features else None

        x = self.adain1(x5, z)
        x = self.up1(x, x4)
        if return_features:
            features.append(x)
        x = self.adain2(x, z)
        x = self.up2(x, x3)
        if return_features:
            features.append(x)
        x = self.adain3(x, z)
        x = self.up3(x, x2)
        if return_features:
            features.append(x)
        x = self.adain4(x, z)
        x = self.up4(x, x1)
        if return_features:
            features.append(x)
        logits = self.outc(x)
        if return_features:
            return logits, features
        return logits


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = CUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        force_fp32=False,
        return_features=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        out = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            return_features=return_features,
            **model_kwargs,
        )

        if return_features:
            logits, features = out
        else:
            logits = out
            features = None

        assert logits.dtype == dtype
        D_x = c_skip * x + c_out * logits.to(torch.float32)

        if return_features:
            return D_x, features
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

# exp_small_disc_10k
# class DiscriminatorHead(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(in_channels, 1),
#         )

#     def forward(self, x):
#         return self.net(x) #.view(-1)


class DiscriminatorHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels * 4, in_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels * 4, 1),
        )

    def forward(self, x):
        return self.net(x)


class DiscHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels * 4, in_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels * 4, 1),
        )

    def forward(self, x):
        return self.net(x)
