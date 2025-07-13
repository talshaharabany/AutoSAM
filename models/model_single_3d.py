from models.hardnet import HarDNet
from models.base import *


class Decoder(nn.Module):
    def __init__(self, full_features, out):
        super(Decoder, self).__init__()
        # self.up1 = UpBlockSkip(full_features[4] + full_features[3], full_features[3],
        #                        func='relu', drop=0).cuda()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).cuda()
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).cuda()
        self.up3 = UpBlockSkip(full_features[1] + full_features[0], full_features[0],
                               func='relu', drop=0).cuda()
        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.final = CNNBlock(full_features[0], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        z = self.up3(z, x[0])
        # z = self.up4(z, x[0])
        z = self.Upsample(z)
        out = F.tanh(self.final(z))
        return out


class Unet(nn.Module):
    def __init__(self, order, depth_wise, args):
        super(Unet, self).__init__()
        self.backbone = HarDNet(depth_wise=depth_wise, arch=order, args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = Decoder(d, out=1)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img, size=None):
        z = self.backbone(img)
        M = self.decoder(z)
        return M


# class SmallDecoder(nn.Module):
#     def __init__(self, full_features, out):
#         super(SmallDecoder, self).__init__()
#         self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
#                                func='relu', drop=0).cuda()
#         self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
#                                func='relu', drop=0).cuda()
#         self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)
#
#     def forward(self, x):
#         z = self.up1(x[3], x[2])
#         z = self.up2(z, x[1])
#         out = F.tanh(self.final(z))
#         return out


class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoder, self).__init__()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = F.tanh(self.final(z))
        # out = self.final(z)
        return out


class SparseDecoder(nn.Module):
    def __init__(self, full_features, out, nP):
        super(SparseDecoder, self).__init__()
        self.final = CNNBlock(full_features[-1], 256, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.final(x[-1])
        out = z.reshape(8, 256, -1).permute(0, 2, 1)
        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        nP = int(args['nP']) + 1
        half = 0.5*nP**2
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = Decoder(d, out=4)
        for param in self.backbone.parameters():
            param.requires_grad = True
        x = torch.arange(nP, nP**2, nP).long()
        y = torch.arange(nP, nP**2, nP).long()
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        P = torch.cat((grid_x.unsqueeze(dim=0), grid_y.unsqueeze(dim=0)), dim=0)
        P = P.view(2, -1).permute(1, 0).cuda()
        self.P = (P - half) / half
        pos_labels = torch.ones(P.shape[-2])
        neg_labels = torch.zeros(P.shape[-2])
        self.labels = torch.cat((pos_labels, neg_labels)).cuda().unsqueeze(dim=0)

    def forward(self, img, size=None):
        if size is None:
            half = img.shape[-1] / 2
        else:
            half = size / 2
        P = self.P.unsqueeze(dim=0).repeat(img.shape[0], 1, 1).unsqueeze(dim=1)
        z = self.backbone(img)
        J = self.decoder(z)
        dPx_neg = F.grid_sample(J[:, 0:1], P).transpose(3, 2)
        dPx_pos = F.grid_sample(J[:, 2:3], P).transpose(3, 2)
        dPy_neg = F.grid_sample(J[:, 1:2], P).transpose(3, 2)
        dPy_pos = F.grid_sample(J[:, 3:4], P).transpose(3, 2)
        dP_pos = torch.cat((dPx_pos, dPy_pos), -1)
        dP_neg = torch.cat((dPx_neg, dPy_neg), -1)
        P_pos = dP_pos + P
        P_neg = dP_neg + P
        P_pos = P_pos.clamp(min=-1, max=1)
        P_neg = P_neg.clamp(min=-1, max=1)
        points_norm = torch.cat((P_pos, P_neg), dim=2)
        points = (points_norm * half) + half
        return points, self.labels, J, points_norm


class ModelEmb(nn.Module):
    def __init__(self, args):
        super(ModelEmb, self).__init__()
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SmallDecoder(d, out=256)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img, size=None):
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        return dense_embeddings


class ModelSparseEmb(nn.Module):
    def __init__(self, args):
        super(ModelSparseEmb, self).__init__()
        nP = int(args['nP'])
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SparseDecoder(d, out=1, nP=nP)
        for param in self.backbone.parameters():
            param.requires_grad = True
        # pos_labels = torch.ones(int(args['nP']))
        # neg_labels = torch.zeros(int(args['nP']))
        # self.labels = torch.cat((pos_labels, neg_labels)).cuda().unsqueeze(dim=0)

    def forward(self, img, size=None):
        z = self.backbone(img)
        sparse_embeddings = self.decoder(z)
        return sparse_embeddings

'''
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
'''

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the mean and variance over the spatial dimensions (depth, height, width)
        mean = x.mean(dim=[2, 3, 4], keepdim=True)  # Mean across depth, height, width
        var = ((x - mean) ** 2).mean(dim=[2, 3, 4], keepdim=True)  # Variance across depth, height, width
        
        # Normalize the input
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply the scale and bias (weight and bias)
        x = self.weight.view(1, -1, 1, 1, 1) * x + self.bias.view(1, -1, 1, 1, 1)
        
        return x



class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=2, stride=2)
        self.norm1 = LayerNorm2d(4)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv3d(4, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv3d(16, 256, kernel_size=1)

    def forward(self, mask):
        z = self.conv1(mask)
        z = self.norm1(z)
        z = self.gelu(z)
        z = self.conv2(z)
        z = self.norm2(z)
        z = self.gelu(z)
        z = self.conv3(z)
        return z


class ModelH(nn.Module):
    def __init__(self):
        super(ModelH, self).__init__()
        self.conv1 = nn.ConvTranspose3d(256, 64, 3, stride=2, padding=1)
        self.norm1 = LayerNorm2d(64)
        self.gelu = nn.GELU()
        self.conv2 = nn.ConvTranspose3d(64, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, mask):
        z = self.conv1(mask, output_size=(128, 128))
        z = self.norm1(z)
        z = self.gelu(z)
        z = self.conv2(z, output_size=(256, 256))
        z = self.norm2(z)
        z = self.gelu(z)
        z = self.conv3(z)
        return z


if __name__ == "__main__":
    import argparse
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-nP', '--nP', default=10, help='image size', required=False)
    args = vars(parser.parse_args())

    # sam_args = {
    #     'sam_checkpoint': "/home/tal/MedicalSam/cp/sam_vit_h_4b8939.pth",
    #     'model_type': "vit_h",
    #     'generator_args': {
    #         'points_per_side': 8,
    #         'pred_iou_thresh': 0.95,
    #         'stability_score_thresh': 0.7,
    #         'crop_n_layers': 0,
    #         'crop_n_points_downscale_factor': 2,
    #         'min_mask_region_area': 0,
    #         'point_grids': None,
    #         'box_nms_thresh': 0.7,
    #     },
    #     'gpu_id': 0,
    # }

    model = ModelH().cuda()
    # x = torch.randn((3, 3, 256, 256)).cuda()
    # P = model(x)
    # sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    # sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    # pretrain = sam.prompt_encoder.mask_downscaling

    # model = MaskEncoder().cuda()
    # model.conv1.load_state_dict(pretrain[0].state_dict())
    # model.norm1.load_state_dict(pretrain[1].state_dict())
    # model.conv2.load_state_dict(pretrain[3].state_dict())
    # model.norm2.load_state_dict(pretrain[4].state_dict())
    # model.conv3.load_state_dict(pretrain[6].state_dict())
    x = torch.randn((4, 256, 64, 64)).cuda()
    z = model(x)
    print(z.shape)
