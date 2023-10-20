# Modified from https://github.com/luca-medeiros/lang-segment-anything/blob/main/lang_sam/lang_sam.py
import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    """
    Load a model using HuggingFace Hub and configuration files.

    Args:
        repo_id (str): Hugging Face hub repository ID.
        filename (str): Model checkpoint filename.
        ckpt_config_filename (str): Model configuration filename.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: Loaded model.
    """
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    """
    Transform an image for model input.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        torch.Tensor: Transformed image as a tensor.
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class LangSAM():
    """
    LangSAM class for language-guided image segmentation.

    Args:
        sam_type (str): Type of SAM model to use. ["vit_h", "vit_l", "vit_b"]. Default as "vit_h".
        ckpt_path (str): Path to a custom checkpoint for the SAM model.
    """
    def __init__(self, sam_type="vit_h", ckpt_path=None):
        self.sam_type = sam_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_sam(ckpt_path)

    def build_sam(self, ckpt_path):
        if self.sam_type is None or ckpt_path is None:
            if self.sam_type is None:
                print("No sam type indicated. Using vit_h by default.")
                self.sam_type = "vit_h"
            checkpoint_url = SAM_MODELS[self.sam_type]
            try:
                sam = sam_model_registry[self.sam_type]()
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                sam.load_state_dict(state_dict, strict=True)
            except:
                raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                    and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                    re-downloading it.")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
        else:
            try:
                sam = sam_model_registry[self.sam_type](ckpt_path)
            except:
                raise ValueError(f"Problem loading SAM. Your model type: {self.sam_type} \
                should match your checkpoint path: {ckpt_path}. Recommend calling LangSAM \
                using matching model type AND checkpoint path")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        """
        Perform object detection using the GroundingDINO model.

        Args:
            image_pil (PIL.Image.Image): Input image.
            text_prompt (str): Text prompt for grounding objects.
            box_threshold (float): Box confidence threshold.
            text_threshold (float): Text confidence threshold.

        Returns:
            torch.Tensor: Predicted boxes.
            torch.Tensor: Logits for text.
            torch.Tensor: Predicted text phrases.
        """
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        """
        Perform image segmentation using the SAM model.

        Args:
            image_pil (PIL.Image.Image): Input image.
            boxes (torch.Tensor): Predicted boxes from GroundingDINO.

        Returns:
            torch.Tensor: Predicted segmentation masks.
        """
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Perform language-guided image segmentation.

        Args:
            image_pil (PIL.Image.Image): Input image.
            text_prompt (str): Text prompt for grounding objects.
            box_threshold (float): Box confidence threshold.
            text_threshold (float): Text confidence threshold.

        Returns:
            torch.Tensor: Predicted segmentation masks.
            torch.Tensor: Predicted boxes.
            torch.Tensor: Predicted text phrases.
            torch.Tensor: Logits for text.
        """
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits

    def predict_adaptive(self, image_pil, text_prompt, box_threshold=0.1, text_threshold=0.1, step=0.05):
        """
        Adaptive prediction to prevent empty mask results.

        Args:
            image_pil (PIL.Image.Image): Input image.
            text_prompt (str): Text prompt for grounding objects.
            box_threshold (float): Box confidence threshold.
            text_threshold (float): Text confidence threshold.
            step (float): Step size for increasing thresholds.

        Returns:
            torch.Tensor: Predicted segmentation masks.
            torch.Tensor: Predicted boxes.
            torch.Tensor: Predicted text phrases.
            torch.Tensor: Logits for text.
        """
        masks = torch.tensor([])
        while(masks.numel() == 0):
            masks, boxes, phrases, logits = self.predict(image_pil, text_prompt, box_threshold, text_threshold)
            box_threshold += step
            text_threshold += step
            print("masks: ", masks.shape)
        return masks, boxes, phrases, logits

#https://github.com/luca-medeiros/lang-segment-anything/blob/main/lang_sam/utils.py
def draw_image(image, masks, boxes, labels, alpha=0.4):
    """
    Draw boxes and segmentation masks on an image.

    Args:
        image (np.ndarray): Input image.
        masks (torch.Tensor): Predicted segmentation masks.
        boxes (torch.Tensor): Predicted boxes.
        labels (List[str]): Labels for boxes.
        alpha (float): Transparency for segmentation masks.

    Returns:
        np.ndarray: Image with drawn boxes and masks.
    """
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)
