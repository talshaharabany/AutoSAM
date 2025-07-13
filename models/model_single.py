from models.hardnet import HarDNet
from models.base import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18  # 3D ResNet
import time
from torchvision.models.video import R3D_18_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(self, full_features, out):
        super(Decoder, self).__init__()
        # self.up1 = UpBlockSkip(full_features[4] + full_features[3], full_features[3],
        #                        func='relu', drop=0).to(device)
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).to(device)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).to(device)
        self.up3 = UpBlockSkip(full_features[1] + full_features[0], full_features[0],
                               func='relu', drop=0).to(device)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
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


class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoder, self).__init__()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0).to(device)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0).to(device)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = F.tanh(self.final(z))
        return out

class MaskRefinement2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskRefinement2D, self).__init__()
        # A simple 2D convolution layer with a sigmoid to smooth and rescale the mask
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)  # Convolution to refine the mask
        x = self.sigmoid(x)  # Apply sigmoid to output values between [0, 1]
        return x


class SmallDecoder3D(nn.Module):
    def __init__(self, out):
        super(SmallDecoder3D, self).__init__()
        # According to r3d_18:
        # x[0] - layer1: 64 channels
        # x[1] - layer2: 128 channels
        # x[2] - layer3: 256 channels
        # x[3] - layer4: 512 channels
        
        self.up1 = UpBlockSkip3D(512 + 256, 256, func='relu', drop=0)
        self.up2 = UpBlockSkip3D(256 + 128, 128, func='relu', drop=0)
        self.final = CNNBlock3D(128, out, kernel_size=3, drop=0)

        # 🔧 Custom weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x[::-1]
        z = self.up1(x[0], x[1])
        z = self.up2(z, x[2])
        out = self.final(z)
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
        P = P.view(2, -1).permute(1, 0).to(device)
        self.P = (P - half) / half
        pos_labels = torch.ones(P.shape[-2])
        neg_labels = torch.zeros(P.shape[-2])
        self.labels = torch.cat((pos_labels, neg_labels)).to(device).unsqueeze(dim=0)

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


class ModelEmb3D(nn.Module):
    def __init__(self, args):
        super(ModelEmb3D, self).__init__()
        full_backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)

        self.backbone = nn.Sequential(
            full_backbone.stem,
            full_backbone.layer1,
            full_backbone.layer2,
            full_backbone.layer3,
            full_backbone.layer4
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False  

        self.decoder = SmallDecoder3D(out=1)
        for param in self.decoder.parameters():
            param.requires_grad = True

        if torch.cuda.is_available():
            self.backbone = self.backbone#.half()
            self.decoder = self.decoder#.half()  # Keep decoder in full precision for stability
            self.Idim = int(args['Idim'])
            self.NumSliceDim = int(args['NumSliceDim'])

    def forward(self, img):
        if img.device.type == 'cuda':
            img = img#.half()

        if img.shape[2] > 32 or img.shape[3] > 256 or img.shape[4] > 256:
            print(f"⚠ Warning: Very large input detected: {img.shape}. This may be very slow.")

        z = []
        x = img
        start_all = time.time()
        for idx, layer in enumerate(self.backbone):
            start_layer = time.time()
            x = layer(x)
            z.append(x)


        start_decoder = time.time()
        # FIX: pass correct order of feature maps to decoder
        dense_embeddings = self.decoder(z)

        # assert dense_embeddings.requires_grad, "❌ Decoder output does not require grad!"

        dense_embeddings = F.interpolate(
            dense_embeddings, size=(self.NumSliceDim, self.Idim, self.Idim), mode='trilinear', align_corners=True
        )

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
        # self.labels = torch.cat((pos_labels, neg_labels)).to(device).unsqueeze(dim=0)

    def forward(self, img, size=None):
        z = self.backbone(img)
        sparse_embeddings = self.decoder(z)
        return sparse_embeddings


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


class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.norm1 = LayerNorm2d(4)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv2d(16, 256, kernel_size=1)

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
        self.conv1 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1)
        self.norm1 = LayerNorm2d(64)
        self.gelu = nn.GELU()
        self.conv2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)

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

    model = ModelH().to(device)
    # x = torch.randn((3, 3, 256, 256)).to(device)
    # P = model(x)
    # sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    # sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    # pretrain = sam.prompt_encoder.mask_downscaling

    # model = MaskEncoder().to(device)
    # model.conv1.load_state_dict(pretrain[0].state_dict())
    # model.norm1.load_state_dict(pretrain[1].state_dict())
    # model.conv2.load_state_dict(pretrain[3].state_dict())
    # model.norm2.load_state_dict(pretrain[4].state_dict())
    # model.conv3.load_state_dict(pretrain[6].state_dict())
    x = torch.randn((4, 256, 64, 64)).to(device)
    z = model(x)
    print(z.shape)




