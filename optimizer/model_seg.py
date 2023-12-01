# from models.vanilla import *
from models.hardnet import *
from models.resnet import *
from models.base import *
from hard_arch import *


class Segmentation(nn.Module):
    def __init__(self, args):
        super(Segmentation, self).__init__()
        if args['backbone'] == 'resnet':
            self.E = ResNet(args=args)
        if args['backbone'] == 'hardnet':
            self.E = HardNet_model(args=args)
        self.D = SkipDecoder(self.E.full_features,
                             out_channel=1)

    def forward(self, I):
        size = I.size()[2:]
        z = self.E(I)
        return F.sigmoid(self.D(z, size))














