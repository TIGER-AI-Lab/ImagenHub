import torch
from PIL import Image
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import inference_seem_pano

from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
import os
from PIL import Image

def resolve_path(path):
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, path)

class SoM:
    def __init__(
        self,
        semsam_cfg="configs/semantic_sam_only_sa-1b_swinL.yaml",
        seem_cfg="configs/seem_focall_unicl_lang_v1.yaml",
        semsam_ckpt="../../../checkpoint/SoM/swinl_only_sam_many2many.pth",
        sam_ckpt="../../../checkpoint/SoM/sam_vit_h_4b8939.pth",
        seem_ckpt="../../../checkpoint/SoM/seem_focall_v1.pt"
    ):
        semsam_ckpt = resolve_path(semsam_ckpt)
        sam_ckpt = resolve_path(sam_ckpt)
        seem_ckpt = resolve_path(seem_ckpt)
        # Load options and initialize distributed
        self.opt_semsam = load_opt_from_config_file(resolve_path(semsam_cfg))
        self.opt_seem = load_opt_from_config_file(resolve_path(seem_cfg))
        self.opt_seem = init_distributed_seem(self.opt_seem)

        # Load models
        self.model_semsam = BaseModel(self.opt_semsam, build_model(self.opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
        self.model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
        self.model_seem = BaseModel_Seem(self.opt_seem, build_model_seem(self.opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

        # Preload text embeddings for SEEM
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                self.model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
                )

    @torch.no_grad()
    def add_marks(self, image_path, method='semantic-sam', alpha=0.1, label_mode='Number', anno_mode=['Mask', 'Mark']):
        img = Image.open(image_path).convert("RGB")

        label_mode_internal = 'a' if label_mode == 'Alphabet' else '1'

        if method == 'semantic-sam':
            output, mask = inference_semsam_m2m_auto(
                self.model_semsam, img, level=[4], all_classes=None, thresh=None, all_parts=None,
                text_size=400, hole_scale=100, island_scale=100,
                semantic=False, label_mode=label_mode_internal, alpha=alpha, anno_mode=anno_mode
            )
        elif method == 'sam':
            output, mask = inference_sam_m2m_auto(
                self.model_sam, img, text_size=800, label_mode=label_mode_internal, alpha=alpha, anno_mode=anno_mode
            )
        elif method == 'seem':
            output, mask = inference_seem_pano(
                self.model_seem, img, text_size=400, label_mode=label_mode_internal, alpha=alpha, anno_mode=anno_mode
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return Image.fromarray(output) 
