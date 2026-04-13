import torch
import torch.nn as nn


# ------------------ DOUBLE CONV BLOCK ------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Dropout2d(0.2)  # improves generalization
        )

    def forward(self, x):
        return self.conv(x)


# ------------------ U-NET ------------------
class UNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(1, 48)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(48, 96)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(96, 192)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(192, 384)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(384, 768)

        # Decoder
        self.up4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(768, 384)

        self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(384, 192)

        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(192, 96)

        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(96, 48)

        # Final layer
        self.final = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        # Bottleneck
        b = self.bottleneck(self.pool4(d4))

        # Decoder + Skip Connections
        u4 = self.up4(b)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.final(u1)

        