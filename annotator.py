import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def sam_call(image, sam, points, original_size):
    image_tensor = torch.tensor(image).permute(2,0,1)
    image_tensor = transform.apply_image_torch(image_tensor)
    input_images = transform.preprocess(image_tensor).unsqueeze(dim=0)
    with torch.no_grad():
        image_embeddings = sam.image_encoder(input_images)
        point_coords = transform.apply_coords(np.array(points), original_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device='cpu').unsqueeze(dim=0)
        labels_torch = torch.as_tensor([1], dtype=torch.int, device='cpu')
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        points = (coords_torch, labels_torch)
        sparse_embeddings, dense_embeddings_none = sam.prompt_encoder(points=points, boxes=None, masks=None)
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings_none,
            multimask_output=False,
        )
    return low_res_masks, image_tensor


def segment_image(image, point):
    original_size = image.shape[:2]
    mask, image_tensor = sam_call(image, sam, point, original_size)
    input_size = image_tensor.shape[1:]
    mask = sam.postprocess_masks(mask, input_size=input_size, original_size=original_size)
    mask = mask.squeeze().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (255*mask).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(mask, point, 6, (0, 0, 255), -1)
    return mask

def click_event(event, x, y, flags, param):
    global image, point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        mask = segment_image(image, point)
        cv2.imwrite('tmp.jpg', mask)
        cv2.imshow("Mask", mask)


if __name__ == "__main__":
    # Load your image
    image = cv2.imread("me.png")
    point = None
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)
    transform = ResizeLongestSide(1024)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
