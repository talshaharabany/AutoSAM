import torch
import torch.nn as nn
from torchvision import models


class VGG16Net(nn.Module):
    def __init__(self):
        super(VGG16Net, self).__init__()
        self.Org_model = models.vgg16(pretrained=True).features[:23]
        self.full_features = [64, 128, 256, 512, 512]
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        for i in range(23):
            x = self.Org_model[i](x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.Org_model = models.vgg16(pretrained=True).features[:23]
        self.full_features = [64, 128, 256, 512, 512]
        for param in self.Org_model.parameters():
            param.requires_grad = True
        self.layers = [3, 4, 8, 15, 22]
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = []
        for i in range(23):
            x = self.Org_model[i](x)
        return x


if __name__ == "__main__":
    model = VGG16Net().cuda()
    x = torch.randn((16, 3, 304, 304)).cuda()
    z = model(x)
    print(z.shape)
    # for item in z:
    #     print(item.shape)
