from models.vanilla import *
import neural_renderer as nr
from models.hardnet import *
from utils.utils_train import norm_input

class Decoder(nn.Module):
    def __init__(self, full_features, args):
        super(Decoder, self).__init__()
        if int(args['outlayer']) == 2:
            self.up1 = UpBlock(full_features[1] + full_features[0], 2,
                               func='tanh', drop=float(args['drop'])).cuda()
        if int(args['outlayer']) == 3:
            self.up1 = UpBlock(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=float(args['drop'])).cuda()
            self.up2 = UpBlock(full_features[1] + full_features[0], 2,
                               func='tanh', drop=float(args['drop'])).cuda()
        if int(args['outlayer']) == 4:
            self.up1 = UpBlock(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=float(args['drop'])).cuda()
            self.up2 = UpBlock(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=float(args['drop'])).cuda()
            self.up3 = UpBlock(full_features[1] + full_features[0], 2,
                               func='tanh', drop=float(args['drop'])).cuda()
        self.args = args

    def forward(self, x, size):
        if int(self.args['outlayer']) == 2:
            shift_map = self.up1(x[1], x[0])
        if int(self.args['outlayer']) == 3:
            z = self.up1(x[2], x[1])
            shift_map = self.up2(z, x[0])
        if int(self.args['outlayer']) == 4:
            z = self.up1(x[3], x[2])
            z = self.up2(z, x[1])
            shift_map = self.up3(z, x[0])
        shift_map = F.interpolate(shift_map, size=size, mode='bilinear', align_corners=True)
        return shift_map[:, 0, :, :].unsqueeze(dim=1), shift_map[:, 1, :, :].unsqueeze(dim=1)
     

class DeepACM(nn.Module):
    def __init__(self, args):
        super(DeepACM, self).__init__()
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        self.ACMDecoder = Decoder(self.backbone.full_features, args)
        self.nP = int(args['nP'])
        self.texture_size = 2
        self.camera_distance = 1
        self.elevation = 0
        self.azimuth = 0
        self.image_size = int(args['im_size'])
        self.renderer = nr.Renderer(camera_mode='look_at', image_size=self.image_size, light_intensity_ambient=1,
                                    light_intensity_directional=1, perspective=False)

    def forward(self, I, P, faces, it):
        size = I.size()[2:]
        z = self.backbone(I)
        Ix, Iy = self.ACMDecoder(z, size)
        masks = []
        Ps = []
        for i in range(it):
            Pxx = F.grid_sample(Ix, P).transpose(3, 2)
            Pyy = F.grid_sample(Iy, P).transpose(3, 2)
            Pedge = torch.cat((Pxx, Pyy), -1)
            P = Pedge + P
            z = torch.ones((P.shape[0], 1, P.shape[2], 1)).cuda()
            PP = torch.cat((P, z), 3)
            PP = torch.squeeze(PP, dim=1)
            PP[:, :, 1] = PP[:, :, 1]*-1
            faces = torch.squeeze(faces, dim=1)
            self.renderer.eye = nr.get_points_from_angles(self.camera_distance, self.elevation, self.azimuth)
            mask = self.renderer(PP, faces, mode='silhouettes').unsqueeze(dim=1)
            PP[:, :, 1] = PP[:, :, 1]*-1
            Ps.append(PP[:, :, 0:2].unsqueeze(dim=1))
            masks.append(mask)
        return masks, Ps, Ix, Iy, I














