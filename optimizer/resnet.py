import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        if int(args['order']) == 18:
            self.Org_model = models.resnet18(pretrained=True)
            self.full_features = [64, 64, 128, 256, 512]
        elif int(args['order']) == 34:
            self.Org_model = models.resnet34(pretrained=True)
            self.full_features = [64, 64, 128, 256, 512]
        elif int(args['order']) == 50:
            self.Org_model = models.resnet50(pretrained=True)
            self.full_features = [64, 256, 512, 1024, 2048]
        elif int(args['order']) == 101:
            self.Org_model = models.resnet101(pretrained=True)
            self.full_features = [64, 256, 512, 1024, 2048]
        self.features = self.full_features[-1]
        for param in self.Org_model.parameters():
            param.requires_grad = True
        self.features = self.features

    def forward(self, x):
        x = self.Org_model.conv1(x)
        x = self.Org_model.bn1(x)
        x = self.Org_model.relu(x)
        x1 = self.Org_model.maxpool(x)
        x2 = self.Org_model.layer1(x1)
        x3 = self.Org_model.layer2(x2)
        x4 = self.Org_model.layer3(x3)
        x5 = self.Org_model.layer4(x4)
        return x, x1, x2, x3, x4, x5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-order_ae', '--order_ae', default=50, help='order of the backbone - ae', required=False)
    args = vars(parser.parse_args())

    model = ResNet(args=args).cuda()
    x = torch.randn((16, 3, 224, 224)).cuda()
    z = model(x)
    for i in range(6):
        print(z[i].shape)

