import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0, func=None):
        super(SkipUpBlock, self).__init__()
        d = drop
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(d)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(d)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in, skip, flag=True):
        if flag:
            x_in = self.Upsample(x_in)
        x = torch.cat((x_in, skip), 1)
        x = self.conv1_drop(self.conv1(x))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        if self.func == 'None':
            return x
        elif self.func == 'tanh':
            return F.tanh(self.BN2(x))
        elif self.func == 'sigmoid':
            return F.sigmoid(self.BN2(x))
        elif self.func == 'relu':
            return F.relu(self.BN2(x))
        elif self.func == 'softmax':
            return F.softmax(self.BN2(x), dim=1)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0, func=None):
        super(UpBlock, self).__init__()
        d = drop
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(d)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(d)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in):
        x = self.Upsample(x_in)
        x = self.conv1_drop(self.conv1(x))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        if self.func == 'None':
            return x
        elif self.func == 'tanh':
            return F.tanh(self.BN2(x))
        elif self.func == 'relu':
            return F.relu(self.BN2(x))


class SkipDecoder(nn.Module):
    def __init__(self, full_features, out_channel):
        super(SkipDecoder, self).__init__()
        self.bottleneck = BottleneckBlock(full_features[4], full_features[4])
        self.up1 = SkipUpBlock(full_features[4] + full_features[3], full_features[3],
                               func='relu', drop=0).cuda()
        self.up2 = SkipUpBlock(full_features[3]+full_features[2], full_features[2],
                               func='relu', drop=0).cuda()
        self.up3 = SkipUpBlock(full_features[2]+full_features[1], full_features[1],
                               func='relu', drop=0).cuda()
        self.up4 = SkipUpBlock(full_features[1] + full_features[0], full_features[0],
                               func='relu', drop=0).cuda()
        self.up5 = SkipUpBlock(full_features[0] + full_features[0], out_channel,
                               func='None', drop=0).cuda()

    def forward(self, z, out_size):
        zz = self.bottleneck(z[5])
        zz = self.up1(zz, z[4])
        zz = self.up2(zz, z[3])
        zz = self.up3(zz, z[2])
        zz = self.up4(zz, z[1], False)
        logits = self.up5(zz, z[0])
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=True)


class Decoder(nn.Module):
    def __init__(self, full_features, out_channel=3):
        super(Decoder, self).__init__()
        self.bottleneck = BottleneckBlock(full_features[4], full_features[4])
        self.up0 = UpBlock(full_features[4], full_features[3],
                           func='relu', drop=0).cuda()
        self.up1 = UpBlock(full_features[3], full_features[2],
                           func='relu', drop=0).cuda()
        self.up2 = UpBlock(full_features[2], full_features[1],
                           func='relu', drop=0).cuda()
        self.up3 = UpBlock(full_features[1], full_features[0],
                           func='relu', drop=0).cuda()
        self.up4 = UpBlock(full_features[0], out_channel,
                           func='None', drop=0).cuda()

    def forward(self, z):
        z = self.bottleneck(z)
        z = self.up0(z)
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        return z


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv1_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x_in):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(self.BN2(x))
        return x


