from dataset import transforms_shir as transforms
# from utils import *


def get_cub_transform():
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(22, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    return transform_train, transform_test


def get_glas_transform():
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(5, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    return transform_train, transform_test

# def get_glas_transform():
#     transform_train = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((256, 256)),
#         transforms.ColorJitter(brightness=0.2,
#                                contrast=0.2,
#                                saturation=0.2,
#                                hue=0.1),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomAffine(5, scale=(0.75, 1.25)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
#     ])
#     return transform_train, transform_test


def get_monu_transform(args):
    Idim = int(args['Idim'])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((Idim, Idim)),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(int(args['rotate']), scale=(float(args['scale1']), float(args['scale2']))),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((Idim, Idim)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    return transform_train, transform_test


def get_polyp_transform():
    transform_train = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(90, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    transform_test = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    return transform_train, transform_test
