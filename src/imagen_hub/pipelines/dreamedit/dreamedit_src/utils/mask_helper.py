from PIL import Image
from kornia.morphology import dilation, closing, erosion
from einops import rearrange
import torch
import numpy as np
import torchvision.transforms as T


def merge_masks(masks_tensor):
    """
    param:
        masks in size [N, H, W]
    return:
        mask in size []
    """
    merged_mask, _ = torch.max(masks_tensor, dim=0)
    return merged_mask


def subtract_mask(big_mask, small_mask):
    """
    small_mask must be $\in$ big_mask
    param:
        big_mask in size [H, W]
        small_mask in size [H, W]
    return:
        mask in size [H, W]
    """
    # Subtract the smaller mask from the bigger mask
    result_mask = big_mask - small_mask
    return result_mask


def get_polished_mask(mask_tensor, k: int, mask_type: str):
    """
    param:
        mask in size [H, W]
        k = kernel number
        mask_type name of mask type
    return:
        mask in size [H, W]
    """
    if mask_type == "dilation":
        return mask_dilation(mask_tensor, k)
    elif mask_type == "closing":
        return mask_closing(mask_tensor, k)
    elif mask_type == "closing_half":
        return mask_closing_half(mask_tensor, k)
    else:
        return mask_tensor


def mask_dilation(mask_tensor, k: int):
    """
    param:
        mask in size [H, W]
        k = kernel number
    return:
        mask in size [H, W]
    """
    mask_tensor = rearrange(mask_tensor, 'w h -> 1 1 w h')
    kernel = torch.ones(k, k)
    dilated_img = dilation(mask_tensor, kernel, border_type="constant")
    dilated_img = rearrange(dilated_img, '1 1 w h -> w h')
    return dilated_img


def mask_closing(mask_tensor, k: int):
    """
    param:
        mask in size [H, W]
        k = kernel number
    return:
        mask in size [H, W]
    """
    mask_tensor = rearrange(mask_tensor, 'w h -> 1 1 w h')
    kernel = torch.ones(k, k)
    dilated_img = closing(mask_tensor, kernel, border_type="constant")
    dilated_img = rearrange(dilated_img, '1 1 w h -> w h')
    return dilated_img


def mask_closing_half(mask_tensor, k: int):
    """
    param:
        mask in size [H, W]
        k = kernel number
    return:
        mask in size [H, W]
    """
    mask_tensor = rearrange(mask_tensor, 'w h -> 1 1 w h')
    kernel = torch.ones(k, k)
    dilated_img = dilation(mask_tensor, kernel, border_type="constant")
    kernel = torch.ones(k // 2, k // 2)
    dilated_img = erosion(dilated_img, kernel, border_type="constant")
    dilated_img = rearrange(dilated_img, '1 1 w h -> w h')
    return dilated_img


def transform_box_mask(labeled_box, sam_box, mask):
    mask_box = mask[sam_box[1]:sam_box[3], sam_box[0]:sam_box[2]]
    reshape_y = labeled_box[3] - labeled_box[1]
    reshape_x = labeled_box[2] - labeled_box[0]
    transform = T.Resize((reshape_y, reshape_x))
    mask_box = transform(mask_box.unsqueeze(0))[0]
    mask_return = torch.zeros(mask.shape)
    mask_return[labeled_box[1]:labeled_box[3], labeled_box[0]:labeled_box[2]] = mask_box
    return mask_return


def transform_box_mask_paste(labeled_box, sam_box, mask, background_image, subject_image):
    mask_box = mask[sam_box[1]:sam_box[3], sam_box[0]:sam_box[2]]
    subject_box_array = np.asarray(subject_image)
    subject_box = torch.from_numpy(subject_box_array[sam_box[1]:sam_box[3], sam_box[0]:sam_box[2], :]).permute(2, 0, 1)
    reshape_y = labeled_box[3] - labeled_box[1]
    reshape_x = labeled_box[2] - labeled_box[0]
    transform = T.Resize((reshape_y, reshape_x))
    mask_box = transform(mask_box.unsqueeze(0))[0]
    subject_box = transform(subject_box).permute(1, 2, 0)
    mask_return = torch.zeros(mask.shape)
    subject_copy = torch.zeros(np.asarray(background_image).shape)
    mask_return[labeled_box[1]:labeled_box[3], labeled_box[0]:labeled_box[2]] = mask_box
    subject_copy[labeled_box[1]:labeled_box[3], labeled_box[0]:labeled_box[2], :] = subject_box
    mask_repeat = mask_return.unsqueeze(-1).repeat(1, 1, 3)
    subject_paste = torch.where(mask_repeat > 0, subject_copy, torch.from_numpy(np.asarray(background_image)))
    return mask_return, subject_paste.detach().cpu().numpy()


def resize_box_from_middle(resize_ratio, sam_box):
    y_mid = (sam_box[3] + sam_box[1]) // 2
    x_mid = (sam_box[2] + sam_box[0]) // 2
    y_max = (sam_box[3] - y_mid) * resize_ratio + y_mid
    y_min = y_mid - (sam_box[3] - y_mid) * resize_ratio
    x_max = (sam_box[2] - x_mid) * resize_ratio + x_mid
    x_min = x_mid - (sam_box[2] - x_mid) * resize_ratio
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def resize_box_from_bottom(resize_ratio, sam_box):
    y_mid = sam_box[3]
    x_mid = (sam_box[2] + sam_box[0]) // 2
    y_max = sam_box[3]
    y_min = y_mid - (sam_box[3] - sam_box[1]) * resize_ratio
    x_max = (sam_box[2] - x_mid) * resize_ratio + x_mid
    x_min = x_mid - (sam_box[2] - x_mid) * resize_ratio
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def bounding_box_merge(bbox_tensor):
    x_min_list = bbox_tensor[:, 0].tolist()
    y_min_list = bbox_tensor[:, 1].tolist()
    x_max_list = bbox_tensor[:, 2].tolist()
    y_max_list = bbox_tensor[:, 3].tolist()
    x_min = min(x_min_list)
    y_min = min(y_min_list)
    x_max = max(x_max_list)
    y_max = max(y_max_list)
    result = [int(x_min), int(y_min), int(x_max), int(y_max)]
    result = [max(0, min(512, x)) for x in result]
    return result


def get_mask_from_bbox(bbox):
    return_mask = torch.zeros(512, 512)
    y = bbox[3] - bbox[1]
    x = bbox[2] - bbox[0]
    mask = torch.ones(y, x)
    return_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
    return_mask = return_mask > 0
    return return_mask
