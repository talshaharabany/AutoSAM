

import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import traceback
import random
from torch.cuda.amp import autocast, GradScaler



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from scipy.ndimage import label
from google.colab import drive
import matplotlib.pyplot as plt
import re
from models.model_single import MaskRefinement2D, ModelEmb3D
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from dataset.LungData import get_lung_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import display


class SingleItemDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1e-6):
    alpha = 0.5
    beta = 0.5
    y_pred = y_pred.clamp(0, 1)
    y_true = y_true.clamp(0, 1)

    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3, 4))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3, 4))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3, 4))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict.flatten() + 1
    target = target.flatten() + 1
    tp = np.sum((predict == 2) & (target == 2))
    fp = np.sum((predict == 2) & (target == 1))
    fn = np.sum((predict == 1) & (target == 2))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


# def get_dice_ji(predict, target):
#     predict = predict + 1
#     target = target + 1
#     tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
#     fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
#     fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
#     ji = float(np.nan_to_num(tp / (tp + fp + fn)))
#     dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
#     return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))

def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step, model, debug=True):
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest').float()
    gts_sized = gts_sized.clamp(0, 1)
    masks = masks.clamp(-5, 5)

    if (step + 2) % accumulation_steps == 0 and debug:
        print(f"\n🧪 Step {step} - gts_sized shape: {gts_sized.shape}, masks shape: {masks.shape}")
        print(f"    gts_sized min/max: {gts_sized.min().item():.4f}/{gts_sized.max().item():.4f}")
        print(f"    masks min/max: {masks.min().item():.4f}/{masks.max().item():.4f}")

    # ⛔ No AMP
    loss_1 = criterion(masks, gts_sized)
    loss_2 = Dice_loss(masks.sigmoid(), gts_sized)
    loss = loss_1 + loss_2

    if (step + 2) % accumulation_steps == 0 and debug:
        print(f"    🔁 Loss: BCE = {loss_1.item():.4f}, Dice = {loss_2.item():.4f}, Total = {loss.item():.4f}")

    loss.backward()

    # # Gradient check every 10 steps
    # if (step + 2) % 1 == 0 and debug:
    #     for name, param in model.named_parameters():
    #         # print(f'{name}: {param.dtype}')
    #         if name.startswith("backbone."):
    #             continue
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #                 print(f"⚠ NaN or Inf detected in gradients of: {name}")
    #         else:
    #             print(f"⚠ No gradients for: {name}, requires_grad:{param.requires_grad}")

    # if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()



def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        # if (i==0):
            # print(f"len of images: {len(imgs)}")
            # print(f"img_sz: {len(img_sz)}")
            # print(f"img_sz[i] shape: {img_sz[i].shape}")
            # print(i)
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input


def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).to(device)
    ious = torch.zeros(len(masks_dict)).to(device)
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def train_single_epoch(ds, model, sam, optimizer, transform, epoch):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    optimizer.zero_grad()
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        loss_list.append(loss)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Medical',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
    return np.mean(loss_list)


def train_single_epoch3D(ds, model, sam, optimizer, transform, epoch, mask_refinement):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCEWithLogitsLoss()
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    optimizer.zero_grad()

    consecutive_failures = 0
    max_failures = 10

    save_dir = '/content/drive/MyDrive/segmentation_debug_training'
    os.makedirs(save_dir, exist_ok=True)
    i = 0
    for ix, batch in enumerate(pbar):
        # batch is a list of tuples: (img_tensor, mask_tensor, original_size, img_size, video_path)
        for sample in batch:
            if consecutive_failures >= max_failures:
                print(f"❌ Stopping training after {max_failures} consecutive failures.")
                return np.mean(loss_list) if loss_list else 0

            try:
                img_tensor, gts, original_sz, _ = sample
                orig_img = img_tensor.to(sam.device)
                gts = gts.to(sam.device)

                # Random slice selection:
                volume_depth = orig_img.shape[-1]
                if volume_depth <= NumSliceDim - 1:
                    print(f"⚠ Volume depth {volume_depth} < NumSliceDim {NumSliceDim}, skipping...")
                    continue

                if gts.max() < 0.9:
                    continue
                start_frame = np.random.randint(0, volume_depth - NumSliceDim + 1)
                selected_slices = orig_img[:, :, start_frame:start_frame+NumSliceDim]
                selected_gts = gts[:, :, start_frame:start_frame+NumSliceDim]
                selected_slices = selected_slices.permute(2, 0, 1)
                selected_gts = selected_gts.permute(2, 0, 1).unsqueeze(0)

                if selected_gts.max() < 0.9:
                    continue

                # Resize and adjust
                orig_imgs_small = F.interpolate(
                    selected_slices.unsqueeze(0).unsqueeze(1),  # add batch dim
                    size=(NumSliceDim, Idim, Idim),
                    mode='trilinear', align_corners=True
                )
                if orig_imgs_small.shape[1] == 1:
                    orig_imgs_small = orig_imgs_small.repeat(1, 3, 1, 1, 1)

                # Dense embedding
                dense_embeddings = model(orig_imgs_small)

                for slice_idx in range(NumSliceDim):
                    with torch.cuda.amp.autocast():
                        current_slice = orig_imgs_small[:,:,slice_idx,:,:]
                        current_dense_embeddings = dense_embeddings[:,:,slice_idx,:,:]
                        batched_input = get_input_dict([current_slice], [original_sz], [original_sz])
                        temp_mask = sam_call(batched_input, sam, current_dense_embeddings, mask_refinement).unsqueeze(2)
                    if slice_idx == 0:
                        mask = temp_mask
                    else:
                        mask = torch.cat((mask, temp_mask), 2)

                loss = gen_step(optimizer, selected_gts, mask, criterion, accumulation_steps=4, step=ix, model=model)
                loss_list.append(loss)

                # Reset failures after success
                consecutive_failures = 0

                if (i % 5 == 0):
                    np.savez_compressed(
                        os.path.join(save_dir, f'debug_volume{i}.npz'),
                        image=selected_slices.squeeze().detach().cpu().numpy(),
                        mask=mask.squeeze().detach().cpu().numpy(),
                        gt=selected_gts.squeeze().detach().cpu().numpy()  # 👈 add this line
                    )
                    print("✅ Saved debug volume to Drive")
                i += 1

            except Exception as e:
                print(f"⚠ Exception encountered: {e}, skipping this sample.")
                consecutive_failures += 1
                continue

        pbar.set_description(f'(train) epoch {epoch} :: loss {np.mean(loss_list):.4f}')

    # Save loss array to Google Drive
    try:
        loss_array = np.array(loss_list)
        np.save(f"/content/drive/MyDrive/segmentation_debug/loss_epoch_{epoch}.npy", loss_array)
    except:
        print("!!!!!!!!!!!!failed to save loss array!!!!!!!!!!!!!!")

    return np.mean(loss_list)


def inference_ds(ds, model, sam, transform, epoch, args, mask_refinement):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])

    for img_idx, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        if orig_imgs.ndim == 4:
            orig_imgs = orig_imgs.unsqueeze(0)
        elif orig_imgs.ndim == 3:
            orig_imgs = orig_imgs.unsqueeze(0).unsqueeze(0)
        gts = gts.to(sam.device)

        orig_imgs_small = F.interpolate(
            orig_imgs, 
            (NumSliceDim, Idim, Idim), 
            mode='trilinear', 
            align_corners=True
        )
        if orig_imgs_small.shape[1] == 1:
            orig_imgs_small = orig_imgs_small.repeat(1, 3, 1, 1, 1)

        dense_embeddings = model(orig_imgs_small)

        volume_depth = orig_imgs.shape[-1]
        if volume_depth < NumSliceDim:
            print(f"⚠ Volume depth {volume_depth} < NumSliceDim {NumSliceDim}, skipping...")
            continue

        for slice_idx in range(NumSliceDim):
            current_slice = orig_imgs_small[:, :, slice_idx, :, :]
            current_dense_embeddings = dense_embeddings[:, :, slice_idx, :, :]
            batched_input = get_input_dict([current_slice], [original_sz], [original_sz])
            temp_mask = sam_call(batched_input, sam, current_dense_embeddings, mask_refinement).unsqueeze(2)

            if slice_idx == 0:
                mask = temp_mask
            else:
                mask = torch.cat((mask, temp_mask), dim=2)

        # # Post-process and resize GT for fair comparison
        # masks_resized = torch.sigmoid(mask)
        # masks_resized[masks_resized > 0.5] = 1
        # masks_resized[masks_resized <= 0.5] = 0

        gts_resized = F.interpolate(gts.unsqueeze(0), size=mask.shape[2:], mode='nearest').squeeze(0)
        # imgs_resized = F.interpolate(orig_imgs.unsqueeze(0), size=masks_resized.shape[2:], mode='nearest').squeeze(0)

        save_dir = '/content/drive/MyDrive/segmentation_debug'
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(save_dir, f'debug_volume{img_idx}.npz'),
            image=orig_imgs.squeeze().permute(2, 0, 1).detach().cpu().numpy(),
            mask=mask.squeeze().detach().cpu().numpy(),
            gt=gts_resized.squeeze().detach().cpu().numpy()  # 👈 add this line
        )

        print("✅ Saved debug volume to Drive")

        print(f"inside inference!!! masks shape:{mask.shape} gts:{gts.shape}, image:{orig_imgs.shape}")
        dice, ji = get_dice_ji(
            mask.squeeze().detach().cpu().numpy(),
            gts_resized.squeeze().detach().cpu().numpy()
        )

        iou_list.append(ji)
        dice_list.append(dice)

        pbar.set_description(
            f'(Inference | {args["task"]}) Epoch {epoch} :: Dice {np.mean(dice_list):.4f} :: IoU {np.mean(iou_list):.4f}'
        )

    return np.mean(iou_list)



def sam_call(batched_input, sam, dense_embeddings, mask_refinement):
    with torch.no_grad():
        input_images = sam.preprocess(batched_input[0]["image"])
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    
    # Generate low-res masks (SAM output)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    
    refined_mask = mask_refinement(low_res_masks)  # Apply the refinement layer
    
    return refined_mask


def main(args=None, sam_args=None):

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ModelEmb3D(args=args).float().to(device)
    mask_refinement = MaskRefinement2D(in_channels=1, out_channels=1).float().to(device)

    
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)

    checkpoint = torch.load(sam_args['sam_checkpoint'], weights_only=True, map_location="cpu")
    valid_keys = set(sam.state_dict().keys())
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in valid_keys}
    sam.load_state_dict(filtered_checkpoint, strict=False)
    print("Checkpoint successfully loaded into SAM 2.")

    transform = ResizeLongestSide(1024)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args['learning_rate']),
        weight_decay=float(args['WD'])
    )

    if False:
            # ✅ Set up save directory in Google Drive
        base_save_dir = "/content/drive/My Drive/AutoSAM_results"
        os.makedirs(base_save_dir, exist_ok=True)

        results_dir = os.path.join(base_save_dir, f'gpu1')
        os.makedirs(results_dir, exist_ok=True)

        # ✅ Define best model paths
        args['path_best'] = os.path.join(results_dir, 'net_best.pth')
        path_best = os.path.join(results_dir, 'best.csv')
        args['vis_folder'] = os.path.join(results_dir, 'vis')
        os.makedirs(args['vis_folder'], exist_ok=True)
        model.load_state_dict(torch.load(args['path_best']), strict = False, weights_only = False)
        model.eval()  # Set model to evaluation mode
        IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args, mask_refinement)
        exit()
        

    print('Loading images')
    trainset, testset = get_lung_dataset(args, sam_trans=transform)
    print('Successfully loaded images')

    # For debug mode: use only one sample
    if args.get('debug_mode', False):
        single_sample = trainset[0]
        trainset = SingleItemDataset(single_sample)
        print("⚠ Debug mode: training with a single sample only.")

    ds = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(args['Batch_size']),
        shuffle=True,
        num_workers=int(args['nW']),
        drop_last=True,
        collate_fn=lambda x: x
    )

    
    ds_val = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args['nW_eval']),
        drop_last=False)

    # ✅ Set up save directory in Google Drive
    base_save_dir = "/content/drive/My Drive/AutoSAM_results"
    os.makedirs(base_save_dir, exist_ok=True)

    results_dir = os.path.join(base_save_dir, f'gpu{args["folder"]}')
    os.makedirs(results_dir, exist_ok=True)

    # ✅ Define best model paths
    args['path_best'] = os.path.join(results_dir, 'net_best.pth')
    path_best = os.path.join(results_dir, 'best.csv')
    args['vis_folder'] = os.path.join(results_dir, 'vis')
    os.makedirs(args['vis_folder'], exist_ok=True)

    best = 0
    # path_best = f'results/gpu{args["folder"]}/best.csv'
    f_best = open(path_best, 'w')

    for epoch in range(int(args['epoches'])):
        print(f"Starting epoch {epoch} out of {args['epoches']}")
        train_single_epoch3D(ds, model.train(), sam.eval(), optimizer, transform, epoch, mask_refinement)

        with torch.no_grad():
            IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args, mask_refinement)
            if IoU_val > best:
                torch.save(model.state_dict(), args['path_best'])
                best = IoU_val
                print(f"Best results: {best}")
                f_best.write(f"{epoch},{best}\n")
                f_best.flush()

    f_best.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.01, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=3, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    
    parser.add_argument('-dataset_path', '--dataset_path', default='/content/drive/My Drive/Msc/DeepLearning/Project/Task06_Lung/imagesTr', help='Path to the dataset', required=False)
    parser.add_argument('-mask_path', '--mask_path', default='/content/drive/My Drive/Msc/DeepLearning/Project/Task06_Lung/labelsTr', help='Path to the mask dataset', required=False)

    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=64, help='image size', required=False)
    parser.add_argument('-NumSliceDim', '--NumSliceDim', default=32, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    folder = open_folder('results')
    args['folder'] = folder
    args['path'] = os.path.join('results',
                                'gpu' + folder,
                                'net_last.pth')
    args['path_best'] = os.path.join('results',
                                     'gpu' + folder,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join('results', 'gpu' + args['folder'], 'vis')
    args['debug_mode'] = False#True
    args['visualize'] = True
    os.mkdir(args['vis_folder'])
    # sam_args = {
    #     'sam_checkpoint': "/content/sam2/checkpoints/sam2.1_hiera_large.pt",  # ✅ Choose the correct checkpoint
    #     'config_file': "/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",  # ✅ Correct config file
    #     'vos_optimized': True  # ✅ Enable video segmentation mode
    # }

    sam_args = {
        'sam_checkpoint': "/content/drive/My Drive/sam_vit_h.pth",
        'model_type': "vit_h",
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

    main(args=args, sam_args=sam_args)

