import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> LeakyReLU) * 2 with residual connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.res is None else self.res(x)
        out = self.act1(self.bn1(self.conv1(x)))
        return self.act2(self.bn2(self.conv2(out)) + residual)


class UpBlock(nn.Module):
    """Upscaling then DoubleConv with skip connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class GuidedDepthUNet(nn.Module):
    """
    Lightweight 5-Channel RGB-Guided Depth Upsampling and Refinement U-Net
    Matches exact architecture trained in Google Colab (~3.49M parameters).
    """
    def __init__(self, in_channels=5, out_channels=1, base_filters=32, use_residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual

        # Encoder
        self.inc = DoubleConv(in_channels, base_filters)                                                # 32
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))         # 64
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))     # 128
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))     # 256
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 8, base_filters * 8))# 256

        # Decoder
        self.up1 = UpBlock(base_filters * 16, base_filters * 4)   # 512 -> 128
        self.up2 = UpBlock(base_filters * 8, base_filters * 2)    # 256 -> 64
        self.up3 = UpBlock(base_filters * 4, base_filters)        # 128 -> 32
        self.up4 = UpBlock(base_filters * 2, base_filters)        # 64 -> 32
        
        # Output Refinement Head
        self.outc = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        lr_depth = x[:, 3:4, :, :]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        residual = self.outc(x)
        if self.use_residual:
            return F.relu(lr_depth + residual)
        return F.relu(residual)


class DepthLoss(nn.Module):
    """Masked L1 Loss + Sobel Depth Gradient Loss"""
    def __init__(self, alpha_l1=1.0, alpha_grad=0.5):
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_grad = alpha_grad
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = (gt > 0.05).float()
        else:
            mask = mask.float() * (gt > 0.05).float()
        valid = torch.sum(mask) + 1e-8
        
        l1 = torch.sum(torch.abs(pred - gt) * mask) / valid
        
        gx_p = F.conv2d(pred, self.sobel_x, padding=1)
        gy_p = F.conv2d(pred, self.sobel_y, padding=1)
        gx_g = F.conv2d(gt, self.sobel_x, padding=1)
        gy_g = F.conv2d(gt, self.sobel_y, padding=1)
        
        grad = (torch.sum(torch.abs(gx_p - gx_g) * mask) + torch.sum(torch.abs(gy_p - gy_g) * mask)) / valid
        return self.alpha_l1 * l1 + self.alpha_grad * grad, l1, grad
