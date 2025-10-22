"""
SoM pipeline — single-file mask bundles (methods on class)
------------------------------------------------------------------
- Uses your original SEMSAM / SAM / SEEM wiring.
- Saves **one .npz per image** with all segments.
- Exposes convenient methods on the same object:
    som.save_masks_npz(...)
    som.load_bundle_npz(...)
    som.get_mask_by_label_npz(npz_path, label)
    som.get_masks_from_npz(npz_path, labels, mode="union"|"intersection")

Quick start:

    som = SoM()
    preview, npz_file = som.add_marks(
        slider=1.8,
        image_path="/data/images/cat.jpg",
        save_dir="/tmp/som_npz"
    )

    # Single label
    mask3, meta3 = som.get_mask_by_label_npz(npz_file, 3)
    mask3.save("/tmp/cat_3.png")

    # Union of many
    mu, metas_u = som.get_masks_from_npz(npz_file, [2,5,7], mode="union")
    mu.save("/tmp/cat_2_5_7_union.png")

    # Intersection of many
    mi, metas_i = som.get_masks_from_npz(npz_file, [2,5,7], mode="intersection")
    mi.save("/tmp/cat_2_5_7_intersection.png")
"""

from __future__ import annotations

import os
import io
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import torch

# -----------------------------
# Your original imports
# -----------------------------
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


def resolve_path(path: str) -> str:
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, path)


class SoM:
    """Refined SoM with single-file mask IO helpers as class methods."""

    # =============================================================
    # Construction
    # =============================================================
    def __init__(
        self,
        semsam_cfg: str = "configs/semantic_sam_only_sa-1b_swinL.yaml",
        seem_cfg: str = "configs/seem_focall_unicl_lang_v1.yaml",
        semsam_ckpt: str = "../../../checkpoint/SoM/swinl_only_sam_many2many.pth",
        sam_ckpt: str = "../../../checkpoint/SoM/sam_vit_h_4b8939.pth",
        seem_ckpt: str = "../../../checkpoint/SoM/seem_focall_v1.pt",
    ):
        semsam_ckpt = resolve_path(semsam_ckpt)
        sam_ckpt = resolve_path(sam_ckpt)
        seem_ckpt = resolve_path(seem_ckpt)

        # Load options and initialize distributed
        self.opt_semsam = load_opt_from_config_file(resolve_path(semsam_cfg))
        self.opt_seem = load_opt_from_config_file(resolve_path(seem_cfg))
        self.opt_seem = init_distributed_seem(self.opt_seem)

        # Load models
        self.model_semsam = (
            BaseModel(self.opt_semsam, build_model(self.opt_semsam))
            .from_pretrained(semsam_ckpt)
            .eval()
            .cuda()
        )
        self.model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
        self.model_seem = (
            BaseModel_Seem(self.opt_seem, build_model_seem(self.opt_seem))
            .from_pretrained(seem_ckpt)
            .eval()
            .cuda()
        )

        # Preload text embeddings for SEEM
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
                )

    # =============================================================
    # Inference + optional save (single .npz)
    # =============================================================
    @torch.no_grad()
    def add_marks(
        self,
        slider: float,
        image_path: str,
        method: Optional[str] = None,
        alpha: float = 0.1,
        label_mode: str = "Number",
        anno_mode: List[str] = ["Mask", "Mark"],
        text_size: int = 200,
        save_dir: Optional[str] = None,
        file_suffix: str = "",
    ) -> Tuple[Image.Image, Optional[str]]:
        """
        Runs segmentation and (optionally) saves ONE compact .npz with all segments.

        Returns:
            preview_pil (PIL.Image)
            npz_path or None (if save_dir is None)
        """
        img = Image.open(image_path).convert("RGB")

        # pick model/level (your original logic)
        model_name = method
        if slider < 1.5:
            model_name = "seem"
        elif slider > 2.5:
            model_name = "sam"
        else:
            model_name = "semantic-sam"
            if slider < 1.64:
                level = [1]
            elif slider < 1.78:
                level = [2]
            elif slider < 1.92:
                level = [3]
            elif slider < 2.06:
                level = [4]
            elif slider < 2.20:
                level = [5]
            elif slider < 2.34:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]

        label_mode = "a" if label_mode == "Alphabet" else "1"
        text_size, hole_scale, island_scale = 640, 100, 100
        text, text_part, text_thresh = "", "", "0.0"
        semantic = False

        # run inference (these fns return (overlay_image, anns_list))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if model_name == "semantic-sam":
                overlay_rgb, anns = inference_semsam_m2m_auto(
                    self.model_semsam,
                    img,
                    level,
                    text,
                    text_part,
                    text_thresh,
                    text_size,
                    hole_scale,
                    island_scale,
                    semantic,
                    label_mode=label_mode,
                    alpha=alpha,
                    anno_mode=anno_mode,
                )
            elif model_name == "sam":
                overlay_rgb, anns = inference_sam_m2m_auto(
                    self.model_sam, img, text_size, label_mode, alpha, anno_mode
                )
            else:  # SEEM
                overlay_rgb, anns = inference_seem_pano(
                    self.model_seem, img, text_size, label_mode, alpha, anno_mode
                )

        # optionally save ONE file with everything
        npz_path: Optional[str] = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            npz_path = os.path.join(save_dir, f"{stem}{file_suffix}.npz")
            self.save_masks_npz(npz_path, np.asarray(overlay_rgb), anns)

        return Image.fromarray(overlay_rgb), npz_path

    # =============================================================
    # IO helpers as methods
    # =============================================================
    @staticmethod
    def _to_jpeg_bytes(pil_img: Image.Image, quality: int = 90) -> bytes:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def save_masks_npz(self, npz_path: str, overlay_rgb: np.ndarray, anns: List[Dict[str, Any]]) -> str:
        """
        Save one compact file with:
          - label_map: HxW uint16 with 0=bg, 1..K=segments
          - meta_json: bytes (list of {label, area, bbox, score,...})
          - preview_jpeg: bytes (the overlaid visualization)
        """
        if len(anns) == 0:
            label_map = np.zeros((1, 1), dtype=np.uint16)
            instances: List[Dict[str, Any]] = []
        else:
            h, w = anns[0]["segmentation"].shape
            label_map = np.zeros((h, w), dtype=np.uint16)
            instances = []
            for i, ann in enumerate(anns, start=1):
                seg = ann["segmentation"].astype(bool)
                label_map[seg] = i
                meta_i = {
                    "label": i,
                    "area": int(ann.get("area", int(seg.sum()))),
                    "bbox": ann.get("bbox"),
                    "score": ann.get("predicted_iou", ann.get("score")),
                    "stability_score": ann.get("stability_score"),
                    "id": ann.get("id"),
                }
                instances.append({k: v for k, v in meta_i.items() if v is not None})

        meta_json = json.dumps({"instances": instances}).encode("utf-8")
        preview_jpeg = self._to_jpeg_bytes(Image.fromarray(overlay_rgb))

        np.savez_compressed(
            npz_path,
            label_map=label_map,
            meta_json=np.frombuffer(meta_json, dtype=np.uint8),
            preview_jpeg=np.frombuffer(preview_jpeg, dtype=np.uint8),
        )
        return npz_path

    def load_bundle_npz(self, npz_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[Image.Image]]:
        """Load (label_map, instances_meta_list, preview_image_or_None)."""
        with np.load(npz_path, allow_pickle=False) as z:
            label_map = z["label_map"].astype(np.uint16)
            instances = json.loads(bytes(z["meta_json"]).decode("utf-8")).get("instances", [])
            preview_im = None
            if "preview_jpeg" in z:
                preview_im = Image.open(io.BytesIO(bytes(z["preview_jpeg"])) ).convert("RGB")
        return label_map, instances, preview_im

    def get_mask_by_label_npz(self, npz_path: str, label: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """Return (mask_png_PIL_L_0or255, metadata_dict) for one label."""
        label_map, instances, _ = self.load_bundle_npz(npz_path)
        meta = next((m for m in instances if m.get("label") == label), None)
        if meta is None:
            raise ValueError(f"Label {label} not found in {npz_path}")
        mask = (label_map == np.uint16(label)).astype(np.uint8) * 255
        return Image.fromarray(mask, mode="L")

    def get_masks_from_npz(self, npz_path: str, labels: List[int], mode: str = "union") -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Given a .npz bundle and a list of labels, return:
          - mask_png: PIL Image (L, 8-bit)
          - metas: list of metadata dicts for the selected labels

        mode:
          - "union": pixels belonging to ANY label in the list are 255
          - "intersection": pixels belonging to ALL labels in the list are 255
        """
        label_map, instances, _ = self.load_bundle_npz(npz_path)

        # quick lookup: label -> meta
        meta_by_label = {m["label"]: m for m in instances if "label" in m}

        # collect metas in the requested order (skip missing labels)
        metas = [meta_by_label[l] for l in labels if l in meta_by_label]

        if len(labels) == 0:
            mask = np.zeros_like(label_map, dtype=np.uint8)
        elif mode == "union":
            mask = np.isin(label_map, labels).astype(np.uint8) * 255
        elif mode == "intersection":
            inter = np.ones_like(label_map, dtype=bool)
            for l in labels:
                inter &= (label_map == np.uint16(l))
            mask = inter.astype(np.uint8) * 255
        else:
            raise ValueError("mode must be 'union' or 'intersection'")

        return Image.fromarray(mask, mode="L")
