import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from dataset.tfs import get_lung_transform
import cv2
import nibabel as nib
from scipy.ndimage import zoom
from scipy.ndimage import label
import matplotlib.pyplot as plt
from tqdm import tqdm

def cv2_loader(path, is_mask):
    if is_mask:
        img = cv2.imread(path, 0)
        img[img > 0] = 1
    else:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img

def nib_loader(path, is_mask):
    if is_mask:
        img = nib.load(path).get_fdata()
        img = np.where(img > 0.1, 1, 0).astype(np.float32)
    else:
        img = nib.load(path).get_fdata()
    return img


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=nib_loader,
                 sam_trans=None, loops=1):
        self.root = root
        if train:
            self.imgs_root = os.path.join(self.root, 'Training', 'img')
            self.masks_root = os.path.join(self.root, 'Training', 'mask')
        else:
            self.imgs_root = os.path.join(self.root, 'Testing', 'img')
            self.masks_root = os.path.join(self.root, 'Testing', 'mask')
        self.paths = os.listdir(self.imgs_root)
        self.mask_paths = os.listdir(self.masks_root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.loops = loops
        self.sam_trans = sam_trans
        # self.all_slices = self.preload_all_slices()
        self.all_volumes = self.preload_all_volumes_as_is(train=train)
        print('num of data:{}'.format(len(self.paths)))
        print(f'Number of slices in the dataset: {len(self.all_volumes)}')


    def chunk_volume(self, volume, chunk_depth):
        """
        Chunk a 3D volume into smaller chunks along the depth axis.
        If the volume depth is not divisible by chunk_depth, it will pad with zeros.

        Args:
            volume (numpy.ndarray): The input 3D volume (H, W, D).
            chunk_depth (int): The depth of each chunk.

        Returns:
            List[numpy.ndarray]: List of chunks, each with shape (H, W, chunk_depth).
        """
        depth = volume.shape[2]
        pad_size = (chunk_depth - depth % chunk_depth) % chunk_depth  # calculate needed padding

        if pad_size > 0:
            # Pad along depth with zeros
            volume = np.pad(volume, ((0, 0), (0, 0), (0, pad_size)), mode='constant')

        chunks = []
        for start_idx in range(0, volume.shape[2], chunk_depth):
            chunk = volume[:, :, start_idx:start_idx + chunk_depth]
            chunks.append(chunk)

        return chunks

    def save_volume_as_video(self,volume_tensor, save_path):
        """
        Save a 3D volume (tensor of shape [D, H, W]) as a grayscale MP4 video.
        """
        D, H, W = volume_tensor.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, 4, (W, H), isColor=False)

        for i in range(D):
            frame = volume_tensor[i].cpu().numpy()
            frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)

        video_writer.release()
        print(f"🎥 Saved video at: {save_path}")

    def preload_all_volumes_as_is(self, downscale_factor=(0.5, 0.5, 1.0), train=False):
        if train:
            # cache_dir = '/media/cilab/DATA/Hila/Data/Projects/AutoSAM'
            cache_dir = '/content/drive/My Drive/Projects/AutoSAM'
        else:
            # cache_dir = '/media/cilab/DATA/Hila/Data/Projects/AutoSAM'
            cache_dir = '/content/drive/My Drive/Projects/AutoSAM'

        os.makedirs(cache_dir, exist_ok=True)
        all_volumes = []

        print("Loading full volumes and saving (with existence check):")
        for file_idx, file_path in enumerate(tqdm(self.paths, desc="Loading full volumes")):
            volume_pt_path = os.path.join(cache_dir, f"volume_{file_idx}.pt")

            # ✅ Check if .pt file already exists
            if os.path.exists(volume_pt_path):
                print(f"⚡ Volume {file_idx} already cached. Skipping.")
                volume_data = torch.load(volume_pt_path)
                all_volumes.append(volume_data)
                continue

            # Load and process
            img = self.loader(os.path.join(self.imgs_root, file_path), is_mask=False).astype(np.float32)
            mask = self.loader(os.path.join(self.masks_root, self.mask_paths[file_idx]), is_mask=False).astype(np.float32)
            mask = (mask == 6)
            mask = mask.astype(float)

            # Downscale
            #img = zoom(img, downscale_factor, order=1)
            #mask = zoom(mask, downscale_factor, order=1)
            img = zoom(img, (256 / img.shape[0], 256 / img.shape[1], 92 / img.shape[2]))
            mask = zoom(mask, (256 / mask.shape[0], 256 / mask.shape[1], 92 / mask.shape[2]))

            mask[mask >= 0.5] = 1
            mask[mask <= 0.5] = 0

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # Display the image slice
            axes[0].imshow(img[:,:,50], cmap="gray")
            axes[0].set_title("CT Scan Slice")
            axes[0].axis("off")  # Hide axes

            axes[1].imshow(mask[:,:,50], cmap="gray")
            axes[1].set_title("Segmentation Mask Slice")
            axes[1].axis("off")  # Hide axes
            plt.show()


            img_tensor = torch.tensor(img, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            original_size = torch.tensor(img_tensor.shape)
            img_size = torch.tensor(img_tensor.shape)

            # Save .pt file
            torch.save((img_tensor, mask_tensor, original_size, img_size), volume_pt_path)
            print(f"💾 Saved volume tensor at: {volume_pt_path}")

            all_volumes.append((img_tensor, mask_tensor, original_size, img_size))

        print(f"✅ Done! Total volumes processed or loaded from cache: {len(all_volumes)}")
        return all_volumes

    def preload_all_slices(self):
        all_slices = []
        for file_idx, file_path in enumerate(self.paths):
            # Construct the full path to the image and mask files
            img = self.loader(os.path.join(self.imgs_root, file_path), is_mask=False)
            mask = self.loader(os.path.join(self.masks_root, self.mask_paths[file_idx]), is_mask=False)
            mask = (mask == 6)
            mask = mask.astype(float)

            # Resize the image and mask to 256x256x120
            img = zoom(img, (96 / img.shape[0], 96 / img.shape[1], 96 / img.shape[2]))
            mask = zoom(mask, (96 / mask.shape[0], 96 / mask.shape[1], 96 / mask.shape[2]))

            # Loop over all slices and store them in the all_slices list
            for i in range(img.shape[2]):  # img.shape[2] is the number of slices (z-dimension)
                img_slice = img[:, :, i]  # Get a specific slice
                mask_slice = mask[:, :, i]  # Get the corresponding mask slice

                # Convert to 3-channel (RGB) image
                img_slice = np.stack([img_slice *1/3] * 3, axis=-1)  # (256, 256, 3)

                # Apply transformations
                img_slice, mask_slice = self.transform(img_slice, mask_slice)
                original_size = tuple(img_slice.shape[1:3])  # (256, 256)
                img_slice, mask_slice = self.sam_trans.apply_image_torch(img_slice), self.sam_trans.apply_image_torch(
                    mask_slice)

                # Process mask to binary format
                mask_slice[mask_slice > 0.5] = 1
                mask_slice[mask_slice <= 0.5] = 0


                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                # Display the image slice
                axes[0].imshow(img_slice, cmap="gray")
                axes[0].set_title("CT Scan Slice")
                axes[0].axis("off")  # Hide axes

                axes[1].imshow(mask_slice, cmap="gray")
                axes[1].set_title("Segmentation Mask Slice")
                axes[1].axis("off")  # Hide axes
                plt.show()

                image_size = tuple(img_slice.shape[1:3])  # (256, 256)

                # Store the processed slice and metadata as a tuple
                all_slices.append((self.sam_trans.preprocess(img_slice),
                                   self.sam_trans.preprocess(mask_slice),
                                   torch.Tensor(original_size),
                                   torch.Tensor(image_size)))

        return all_slices



    def __getitem__(self, index):
        img_tensor, mask_tensor, original_size, img_size = self.all_volumes[index % len(self.all_volumes)]
        return img_tensor, mask_tensor, original_size, img_size


#        return self.all_slices[index % len(self.all_slices)]



        '''
        index = index % len(self.paths)
        file_path = self.paths[index]
        mask_path = file_path.split('.')[0] + '.nii.gz'
        img = self.loader(os.path.join(self.imgs_root, file_path), is_mask=False)
        mask = self.loader(os.path.join(self.masks_root, mask_path), is_mask=True)
        img = zoom(img, (256 / img.shape[0], 256 / img.shape[1], 120 / img.shape[2]))
        mask = zoom(mask, (256 / mask.shape[0], 256 / mask.shape[1], 120 / mask.shape[2]))
        dataset_items = []
        img  = np.stack([img[:,:,0]] * 3, axis=-1)
        mask = mask[:,:,0]
        img, mask = self.transform(img, mask)
        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(img), self.sam_trans.apply_image_torch(mask)
        print(img.shape)
        print(mask.shape)
        mask[mask > 0.1] = 1
        mask[mask <= 0.1] = 0
        image_size = tuple(img.shape[1:3])
        return self.sam_trans.preprocess(img), self.sam_trans.preprocess(mask), torch.Tensor(
            original_size), torch.Tensor(image_size)
        '''

    def __len__(self):
        return len(self.paths) * self.loops


def get_lung_dataset(args, sam_trans):
    # datadir = '/media/cilab/DATA/Hila/Data/Projects/AutoSAM/Abdomen'
    datadir = '/content/drive/My Drive/Abdomen'
    transform_train, transform_test = get_lung_transform(args)
    ds_train = ImageLoader(datadir, train=True, transform=transform_train, sam_trans=sam_trans, loops=5)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test, sam_trans=sam_trans)
    return ds_train, ds_test


if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import os
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
        'sam_checkpoint': "../cp/sam_vit_b.pth",
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
    ds_train, ds_test = get_monu_dataset(args, sam_trans)
    ds = torch.utils.data.DataLoader(ds_train,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=True,
                                     drop_last=True)
    pbar = tqdm(ds)
    mean0_list = []
    mean1_list = []
    mean2_list = []
    std0_list = []
    std1_list = []
    std2_list = []
    for i, (img, mask, _, _) in enumerate(pbar):
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

        # a = img.squeeze().permute(1, 2, 0).cpu().numpy()
        # b = mask.squeeze().cpu().numpy()
        # a = (a - a.min()) / (a.max() - a.min())
        # cv2.imwrite('kaki.jpg', 255*a)
        # cv2.imwrite('kaki_mask.jpg', 255*b)