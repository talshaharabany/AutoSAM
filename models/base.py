import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(CNNBlock, self).__init__()
        P = int((kernel_size-1)/2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in, inx=-1):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x_out = self.conv2(x)
        return x_out

class CNNBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(CNNBlock3D, self).__init__()
        P = int((kernel_size-1)/2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout3d(drop)
        self.conv2_drop = nn.Dropout3d(drop)
        self.BN1 = nn.BatchNorm3d(out_channels)
        # self.BN2 = nn.BatchNorm3d(out_channels)

    def forward(self, x_in, inx=-1):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x_out = self.conv2(x)
        return x_out

class UpBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, func=None, drop=0):
        super(UpBlockSkip, self).__init__()
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in, x_up):
        x = self.Upsample(x_in)
        x_cat = torch.cat((x, x_up), 1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if self.func == 'tanh':
            return F.tanh(self.BN(x1))
        elif self.func == 'relu':
            return F.leaky_relu(self.BN(x1))
        elif self.func == 'sigmoid':
            return F.sigmoid(self.BN(x1))
        else:
            return x1

class UpBlockSkip3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, func=None, drop=0):
        super(UpBlockSkip3D, self).__init__()
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout3d(drop)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout3d(drop)
        self.BN = nn.Identity()#nn.BatchNorm3d(out_channels)
        self.func = func

    def forward(self, x_in, x_up):
        x = self.Upsample(x_in)
        x_cat = torch.cat((x, x_up), 1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if self.func == 'tanh':
            return F.tanh(self.BN(x1))
        elif self.func == 'relu':
            return F.leaky_relu(self.BN(x1))
        elif self.func == 'sigmoid':
            return F.sigmoid(self.BN(x1))
        else:
            return x1

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


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(DownBlock, self).__init__()
        P = int((kernel_size -1 ) /2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_in))))
        x1_pool = F.relu(self.BN(self.pool(x1)))
        return x1, x1_pool


class Encoder(nn.Module):
    def __init__(self, AEdim, drop=0):
        super(Encoder, self).__init__()
        self.full_features = [AEdim, AEdim*2, AEdim*4, AEdim*8, AEdim*8]
        self.down1 = DownBlock(3, AEdim, drop=drop)
        self.down2 = DownBlock(AEdim, AEdim*2, drop=drop)
        self.down3 = DownBlock(AEdim*2, AEdim*4, drop=drop)
        self.down4 = DownBlock(AEdim*4, AEdim*8, drop=drop)

    def forward(self, x_in):
        x1, x1_pool = self.down1(x_in)
        x2, x2_pool = self.down2(x1_pool)
        x3, x3_pool = self.down3(x2_pool)
        x4, x4_pool = self.down4(x3_pool)
        return x1, x2, x3, x4, x4_pool


class MMDecoder(nn.Module):
    def __init__(self, full_features, out_channel, z_size, out_size):
        super(MMDecoder, self).__init__()
        self.bottleneck = BottleneckBlock(full_features[4], z_size)
        self.up0 = UpBlock(z_size, full_features[3],
                           func='relu', drop=0).cuda()
        self.up1 = UpBlock(full_features[3], out_channel,
                           func='None', drop=0).cuda()
        self.out_size = out_size

    def forward(self, z, z_text):
        zz = self.bottleneck(z)
        zz_norm = zz / zz.norm(dim=1).unsqueeze(dim=1)
        attn_map = (zz_norm * z_text.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdims=True)
        zz = zz * attn_map
        zz = self.up0(zz)
        zz = self.up1(zz)
        zz = F.interpolate(zz, size=self.out_size, mode="bilinear", align_corners=True)
        return F.sigmoid(zz)


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
