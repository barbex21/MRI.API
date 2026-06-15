import torch
import torch.nn as nn

class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super().__init__()
        self.enc1 = ConvBlock3d(in_channels, init_features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3d(init_features, init_features * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3d(init_features * 2, init_features * 4)

        self.up2 = nn.ConvTranspose3d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(init_features * 4, init_features * 2)
        self.up1 = nn.ConvTranspose3d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(init_features * 2, init_features)

        self.final = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
