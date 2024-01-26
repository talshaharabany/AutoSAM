import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from models.model_single import ModelEmb
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
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
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
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


def inference_ds(ds, model, sam, transform, epoch, args):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    for imgs, gts, original_sz, img_sz in pbar:
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
        masks = sam.postprocess_masks(masks, input_size=input_size, original_size=original_size)
        gts = sam.postprocess_masks(gts.unsqueeze(dim=0), input_size=input_size, original_size=original_size)
        masks = F.interpolate(masks, (Idim, Idim), mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, (Idim, Idim), mode='nearest')
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        dice, ji = get_dice_ji(masks.squeeze().detach().cpu().numpy(),
                               gts.squeeze().detach().cpu().numpy())
        iou_list.append(ji)
        dice_list.append(dice)
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)))
    model.train()
    return np.mean(iou_list)


def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


def main(args=None, sam_args=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ModelEmb(args=args).to(device)
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(args['learning_rate']),
                           weight_decay=float(args['WD']))
    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(args, sam_trans=transform)
    elif args['task'] == 'polyp':
        trainset, testset = get_polyp_dataset(args, sam_trans=transform)
    ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                     num_workers=int(args['nW']), drop_last=True)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    best = 0
    path_best = 'results/gpu' + str(args['folder']) + '/best.csv'
    f_best = open(path_best, 'w')
    for epoch in range(int(args['epoches'])):
        train_single_epoch(ds, model.train(), sam.eval(), optimizer, transform, epoch)
        with torch.no_grad():
            IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args)
            if IoU_val > best:
                torch.save(model, args['path_best'])
                best = IoU_val
                print('best results: ' + str(best))
                f_best.write(str(epoch) + ',' + str(best) + '\n')
                f_best.flush()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=2, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
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
    os.mkdir(args['vis_folder'])
    sam_args = {
        'sam_checkpoint': "cp/sam_vit_b.pth",
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
    main(args=args, sam_args=sam_args)

