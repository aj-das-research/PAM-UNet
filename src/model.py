import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(MBConv, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ProgressiveLuongAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProgressiveLuongAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, 1)
        self.key = nn.Conv2d(in_channels, out_channels, 1)
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        query = self.query(x)
        key = self.key(y)
        value = self.value(y)

        attention = torch.bmm(query.view(query.size(0), -1, query.size(1)),
                              key.view(key.size(0), key.size(1), -1))
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value.view(value.size(0), value.size(1), -1),
                        attention.permute(0, 2, 1))
        out = out.view(value.size())
        out = self.gamma * out + x
        return out

class PAMUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PAMUNet, self).__init__()

        # Encoder
        self.enc1 = MBConv(in_channels, 64, stride=2)
        self.enc2 = MBConv(64, 128, stride=2)
        self.enc3 = MBConv(128, 256, stride=2)
        self.enc4 = MBConv(256, 512, stride=2)

        # Decoder
        self.dec4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att4 = ProgressiveLuongAttention(256, 256)
        self.upconv4 = MBConv(512, 256)

        self.dec3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = ProgressiveLuongAttention(128, 128)
        self.upconv3 = MBConv(256, 128)

        self.dec2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = ProgressiveLuongAttention(64, 64)
        self.upconv2 = MBConv(128, 64)

        self.dec1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = ProgressiveLuongAttention(32, 32)
        self.upconv1 = MBConv(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder
        dec4 = self.dec4(enc4)
        att4 = self.att4(dec4, enc3)
        up4 = self.upconv4(torch.cat([att4, dec4], dim=1))

        dec3 = self.dec3(up4)
        att3 = self.att3(dec3, enc2)
        up3 = self.upconv3(torch.cat([att3, dec3], dim=1))

        dec2 = self.dec2(up3)
        att2 = self.att2(dec2, enc1)
        up2 = self.upconv2(torch.cat([att2, dec2], dim=1))

        dec1 = self.dec1(up2)
        att1 = self.att1(dec1, x)
        up1 = self.upconv1(torch.cat([att1, dec1], dim=1))

        out = self.final_conv(up1)
        return out