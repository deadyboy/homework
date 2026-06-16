import torch
import torch.nn as nn


class FullyConvNetwork(nn.Module):
    """A U-Net style fully convolutional network for paired image translation."""

    def __init__(self):
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.conv1(x)    # (B, 64, 128, 128)
        e2 = self.conv2(e1)   # (B, 128, 64, 64)
        e3 = self.conv3(e2)   # (B, 256, 32, 32)
        e4 = self.conv4(e3)   # (B, 512, 16, 16)
        e5 = self.conv5(e4)   # (B, 1024, 8, 8)

        d5 = self.deconv5(e5)                         # (B, 512, 16, 16)
        d4 = self.deconv4(torch.cat([d5, e4], dim=1))  # (B, 256, 32, 32)
        d3 = self.deconv3(torch.cat([d4, e3], dim=1))  # (B, 128, 64, 64)
        d2 = self.deconv2(torch.cat([d3, e2], dim=1))  # (B, 64, 128, 128)
        output = self.deconv1(torch.cat([d2, e1], dim=1))

        return output
