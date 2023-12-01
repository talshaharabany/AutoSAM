import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        if int(args['order']) == 18:
            self.Org_model = models.resnet18(pretrained=True)
            self.features = 512
            self.full_features = [64, 64, 128, 256, 512]
        elif int(args['order']) == 34:
            self.Org_model = models.resnet34(pretrained=True)
            self.features = 512
            self.full_features = [64, 64, 128, 256, 512]
        elif int(args['order']) == 50:
            self.Org_model = models.resnet50(pretrained=True)
            self.features = 2048
            self.full_features = [64, 256, 512, 1024, 2048]
        elif int(args['order']) == 101:
            self.Org_model = models.resnet101(pretrained=True)
            self.features = 2048
            self.full_features = [64, 256, 512, 1024, 2048]
        for param in self.Org_model.parameters():
            param.requires_grad = True
        layer0 = nn.Sequential(self.Org_model.conv1, self.Org_model.bn1, self.Org_model.relu)
        self.main_layers = [layer0, self.Org_model.layer1, self.Org_model.layer2, self.Org_model.layer3,
                            self.Org_model.layer4]
        self.res = nn.Sequential(*self.main_layers)

    def forward(self, x):
        for inx, layer in enumerate(self.main_layers):
            x = layer(x)
            if inx == 0:
                x2 = x
            elif inx == 1:
                x4 = x
            elif inx == 2:
                x8 = x
            elif inx == 3:
                x16 = x
            else:
                x32 = x
        return x2, x4, x8, x16, x32





