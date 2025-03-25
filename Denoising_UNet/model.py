import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.MaxPool2d(2)
        self.up = Up()
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        self.right1 = DoubleConv(in_channels, 64)
        self.right2 = DoubleConv(64, 128)
        self.right3 = DoubleConv(128, 256)
        self.right4 = DoubleConv(256, 512)
        self.right5 = DoubleConv(512, 1024)
        self.right6 = DoubleConv(1024, 512)
        self.right7 = DoubleConv(512, 256)
        self.right8 = DoubleConv(256, 128)
        self.right9 = DoubleConv(128, 64)

    def forward(self, x):
        x1 = self.right1(x)
        x2 = self.right2(self.down(x1))
        x3 = self.right3(self.down(x2))
        x4 = self.right4(self.down(x3))
        x5 = self.right5(self.down(x4))
        x = self.right6(self.up(x5, x4))
        x = self.right7(self.up(x, x3))
        x = self.right8(self.up(x, x2))
        x = self.right9(self.up(x, x1))
        return self.out(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Up(nn.Module):
    """Upscaling and concatenation block
    x1 -> Features from previous decoder layer
    x2 -> Features from encoder layer(skip connection)
    """

    def __init__(self, bilinear=True, in_channels=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]  # height difference
        diff_x = x2.size()[3] - x1.size()[3]  # width difference

        # F.pad(input, (left, right, top, bottom))

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return x
