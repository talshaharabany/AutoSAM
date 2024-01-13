import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from dataset.tfs import get_polyp_transform
import cv2


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize=352, augmentations=None, train=True, sam_trans=None):
        self.trainsize = trainsize
        self.augmentations = augmentations
        # print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.train = train
        self.sam_trans = sam_trans

    def __getitem__(self, index):
        image = self.cv2_loader(self.images[index], is_mask=False)
        gt = self.cv2_loader(self.gts[index], is_mask=True)
        # image = self.rgb_loader(self.images[index])
        # gt = self.binary_loader(self.gts[index])
        img, mask = self.augmentations(image, gt)
        # mask[mask >= 128] = 255
        # mask[mask < 128] = 0
        # mask[mask == 255] = 1
        # mask = mask.squeeze()
        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(img), self.sam_trans.apply_image_torch(mask)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])
        return self.sam_trans.preprocess(img), self.sam_trans.preprocess(mask), torch.Tensor(
            original_size), torch.Tensor(image_size)
        # return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        # with open(path, 'rb') as f:
            # img = Image.open(f)
            # return img.convert('1')
        img = cv2.imread(path, 0)
        return img

    def cv2_loader(self, path, is_mask):
        if is_mask:
            img = cv2.imread(path, 0)
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        # return 32
        return self.size


def get_polyp_dataset(args, sam_trans=None):
    transform_train, transform_test = get_polyp_transform()
    image_root = 'polyp/TrainDataset/images/'
    gt_root = 'polyp/TrainDataset/masks/'
    ds_train = PolypDataset(image_root, gt_root, augmentations=transform_train, sam_trans=sam_trans)
    image_root = 'polyp/TestDataset/test/images/'
    gt_root = 'polyp/TestDataset/test/masks/'
    ds_test = PolypDataset(image_root, gt_root, train=False, augmentations=transform_test, sam_trans=sam_trans)
    return ds_train, ds_test


def get_tests_polyp_dataset(sam_trans):
    transform_train, transform_test = get_polyp_transform()
    image_root = 'polyp/TestDataset/Kvasir/images/'
    gt_root = 'polyp/TestDataset/Kvasir/masks/'
    ds_Kvasir = PolypDataset(image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    image_root = 'polyp/TestDataset/CVC-ClinicDB/images/'
    gt_root = 'polyp/TestDataset/CVC-ClinicDB/masks/'
    ds_ClinicDB = PolypDataset(image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    image_root = 'polyp/TestDataset/CVC-ColonDB/images/'
    gt_root = 'polyp/TestDataset/CVC-ColonDB/masks/'
    ds_ColonDB = PolypDataset(image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    image_root = 'polyp/TestDataset/ETIS-LaribPolypDB/images/'
    gt_root = 'polyp/TestDataset/ETIS-LaribPolypDB/masks/'
    ds_ETIS = PolypDataset(image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    return ds_Kvasir, ds_ClinicDB, ds_ColonDB, ds_ETIS


# class test_dataset:
#     def __init__(self, image_root, gt_root, testsize=352):
#         self.testsize = testsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.transform = transforms.Compose([
#             transforms.Resize((self.testsize, self.testsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.ToTensor()
#         self.size = len(self.images)
#         self.index = 0
#
#     def load_data(self):
#         image = self.rgb_loader(self.images[self.index])
#         image = self.transform(image).unsqueeze(0)
#         gt = self.binary_loader(self.gts[self.index])
#         name = self.images[self.index].split('/')[-1]
#         if name.endswith('.jpg'):
#             name = name.split('.jpg')[0] + '.png'
#         self.index += 1
#         return image, gt, name
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')


if __name__ == '__main__':
    from tqdm import tqdm
    import argparse
    from matplotlib import pyplot as plt
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    from segment_anything.utils.transforms import ResizeLongestSide

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Idim', '--Idim', default=512, help='learning_rate', required=False)
    parser.add_argument('-pSize', '--pSize', default=4, help='learning_rate', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='learning_rate', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='learning_rate', required=False)
    parser.add_argument('-rotate', '--rotate', default=20, help='learning_rate', required=False)
    args = vars(parser.parse_args())

    sam_args = {
        'sam_checkpoint': "../cp/sam_vit_b_01ec64.pth",
        'model_type': "vit_b",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    sam_trans = ResizeLongestSide(sam.image_encoder.img_size)

    ds_train, ds_test = get_polyp_dataset(args, sam_trans=sam_trans)
    ds = data.DataLoader(dataset=ds_test,
                         batch_size=1,
                         shuffle=False,
                         num_workers=1)
    pbar = tqdm(ds)
    mean0_list = []
    mean1_list = []
    mean2_list = []
    std0_list = []
    std1_list = []
    std2_list = []
    for i, (img, mask, _, _) in enumerate(pbar):
        # img = (img.squeeze().permute(1, 2, 0).cpu().numpy())
        # img = (img - img.min()) / (img.max() - img.min())
        # # mask = mask.numpy()
        #
        # plt.imshow(img)
        # plt.savefig('vis/' + str(i) + ".jpg")
        #
        # blend = 0.5 * mask.permute(1, 2, 0).repeat(1, 1, 3) + 0.5 * img
        # plt.imshow(blend.numpy())
        # plt.savefig('vis/' + str(i) + '_gt.jpg')
        a = img.mean(dim=(0, 2, 3))
        b = img.std(dim=(0, 2, 3))
        mean0_list.append(a[0].item())
        mean1_list.append(a[1].item())
        mean2_list.append(a[2].item())
        std0_list.append(b[0].item())
        std1_list.append(b[1].item())
        std2_list.append(b[2].item())
    print(np.mean(mean0_list))
    print(np.mean(mean1_list))
    print(np.mean(mean2_list))

    print(np.mean(std0_list))
    print(np.mean(std1_list))
    print(np.mean(std2_list))