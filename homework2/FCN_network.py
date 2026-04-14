import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),  # 128 input (64+64 skip)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # 64 input (32+32 skip)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),  # 32 input (16+16 skip)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # 16 input (8+8 skip)
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        e1 = self.conv1(x)    # (B, 8, 128, 128)
        e2 = self.conv2(e1)   # (B, 16, 64, 64)
        e3 = self.conv3(e2)   # (B, 32, 32, 32)
        e4 = self.conv4(e3)   # (B, 64, 16, 16)
        e5 = self.conv5(e4)   # (B, 128, 8, 8)

        # Decoder forward pass with skip connections
        d5 = self.deconv5(e5)                    # (B, 64, 16, 16)
        d4 = self.deconv4(torch.cat([d5, e4], dim=1))  # (B, 32, 32, 32)
        d3 = self.deconv3(torch.cat([d4, e3], dim=1))  # (B, 16, 64, 64)
        d2 = self.deconv2(torch.cat([d3, e2], dim=1))  # (B, 8, 128, 128)
        output = self.deconv1(torch.cat([d2, e1], dim=1))  # (B, 3, 256, 256)

        return output
    