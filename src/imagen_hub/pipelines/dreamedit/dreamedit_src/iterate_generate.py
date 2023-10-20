# modified from https://github.com/DreamEditBenchTeam/DreamEdit/blob/main/src/iterate_generate.py
import warnings

warnings.filterwarnings("ignore")

import os
from omegaconf import OmegaConf
import argparse
import torchvision
import torchvision.transforms as T
import copy
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from imagen_hub.miscmodels.blip import BLIP_Model
from imagen_hub.miscmodels import LangSAM, CLIP, VITs16
from imagen_hub.depend.lang_sam.lang_sam import draw_image
from imagen_hub.metrics.dreambooth_metric import evaluate_dino_score, evaluate_clipi_score, evaluate_clipi_score_list,evaluate_dino_score_list
from imagen_hub.utils.save_image_helper import get_concat_pil_images, get_mask_pil_image, save_pil_image

from .generate import load_model_from_config, diffedit

from .utils.mask_helper import merge_masks, subtract_mask, get_polished_mask, mask_dilation, mask_closing, mask_closing_half, transform_box_mask, transform_box_mask_paste, resize_box_from_middle, resize_box_from_bottom, bounding_box_merge, get_mask_from_bbox
from .pipelines.inpainting_pipelines import select_inpainting_pipeline, inpaint_text_gligen
from .pipelines.extract_object_pipeline import get_object_caption

import json

import logging

logging.getLogger().setLevel(logging.ERROR)

class EncodeRatioScheduler:
    def __init__(self, schedule_config, total_iterations):
        self.start_ratio = schedule_config.start_ratio
        self.end_ratio = schedule_config.end_ratio
        self.decay_type = schedule_config.decay_type
        self.current_ratio = self.start_ratio
        self.manual_ratio_list = []
        if self.decay_type == 'linear':
            self.decay_rate = (self.start_ratio - self.end_ratio) / total_iterations
        elif self.decay_type == 'exponential':
            self.decay_rate = np.power(self.end_ratio / self.start_ratio, 1 / total_iterations)
        elif self.decay_type == 'manual':
            self.decay_rate = 0
            self.manual_ratio_list = copy.deepcopy(schedule_config.manual_ratio_list)
        else:  # constant
            self.decay_rate = 0

    def step(self):
        if self.decay_type == 'linear':
            self.current_ratio -= self.decay_rate
        elif self.decay_type == 'exponential':
            self.current_ratio *= self.decay_rate
        elif self.decay_type == 'manual':
            self.current_ratio = self.manual_ratio_list.pop(0)
        else:  # constant
            self.current_ratio = self.start_ratio
        print("Current encode ratio: ", self.current_ratio)
        return self.current_ratio


def replace_background(
        src_img,
        img_to_edit,
        obj_class,
        first_iter_obj_mask,
        lang_sam_model,
        lang_sam_config,
        de_config,
        difference = False
):
    assert src_img.size == img_to_edit.size
    bg_config = de_config.background_correction
    if bg_config.use_obj_mask_from_first_iteration and first_iter_obj_mask is not None:
        obj_mask = first_iter_obj_mask
        if difference:
            masks, boxes, phrases, logits = lang_sam_model.predict(
                img_to_edit,
                obj_class,
                box_threshold=lang_sam_config.segment_confidence,
                text_threshold=lang_sam_config.segment_confidence,
            )
            obj_mask = merge_masks(masks).long()
            obj_mask = subtract_mask(first_iter_obj_mask, obj_mask).bool() == False
            obj_mask = obj_mask.long()
    else:
        masks, boxes, phrases, logits = lang_sam_model.predict(
            img_to_edit,
            obj_class,
            box_threshold=lang_sam_config.segment_confidence,
            text_threshold=lang_sam_config.segment_confidence,
        )
        obj_mask = merge_masks(masks)
        obj_mask = get_polished_mask(obj_mask, 2, "dilation")
    # mask_config = de_config.mask_config
    # obj_mask = get_polished_mask(obj_mask, mask_config.mask_dilate_kernel, mask_config.mask_type)
    obj_mask = obj_mask.unsqueeze(-1).repeat(1, 1, 3)
    combined = torch.where(
        obj_mask > 0,
        torch.from_numpy(np.asarray(img_to_edit)),
        torch.from_numpy(np.asarray(src_img))
    ).detach().cpu().numpy()
    img_edit = Image.fromarray(np.uint8(combined))
    return img_edit


def iterative_edit(
        src_img_path,  # background image
        img_to_edit_path,
        obj_class,
        lang_sam_model,
        sd_model,
        src_prompt,
        dst_prompt,
        config,
        device,
        labeled_box=None,
):
    src_img = Image.open(src_img_path).resize((512, 512))
    src_img_edited = Image.open(img_to_edit_path).resize((512, 512))
    de_config = config.model.de
    lang_sam_config = config.model.lang_sam
    segment_confidence = lang_sam_config.segment_confidence
    ddim_config = de_config.ddim

    # Step1. Iteratively dream edit
    intermediate_images = []
    obj_mask_list = []
    latents_record_list = []
    generated_image_list = []
    first_iter_obj_mask = None
    encode_ratio_scheduler = EncodeRatioScheduler(ddim_config.encode_ratio_schedule, ddim_config.iteration_number)

    if de_config.use_diffedit:
        pbar = tqdm(range(1))
    else:
        pbar = tqdm(range(ddim_config.iteration_number))
    for i in pbar:
        if de_config.use_diffedit:
            pbar.set_description("Diffedit iteration %s" % i)
        else:
            pbar.set_description("Dream edit iteration %s" % i)

        img_to_edit = Image.open(img_to_edit_path)  # type, PIL image
        if img_to_edit.size != (512, 512):
            img_to_edit = img_to_edit.resize((512, 512))

        # Get object mask
        mask_config = de_config.mask_config
        if (mask_config.use_bbox_mask_for_first_iteration and i == 0) or mask_config.use_bbox_mask_for_all_iterations:
            obj_mask = get_mask_from_bbox(labeled_box).long()

            obj_masks, obj_boxes, phrases, logits = lang_sam_model.predict(
                img_to_edit,
                obj_class,
                box_threshold=segment_confidence,
                text_threshold=segment_confidence,
            )
            obj_mask_org = merge_masks(obj_masks)
            bound_mask = get_mask_from_bbox(labeled_box)
            labeled_img_to_edit = draw_image(
                np.asarray(img_to_edit),
                torch.unsqueeze(bound_mask, 0),
                torch.Tensor([labeled_box]),
                ['label'],
            )
        else:
            # Get the mask of the object from the previous iteration image
            obj_masks, obj_boxes, phrases, logits = lang_sam_model.predict(
                img_to_edit,
                obj_class,
                box_threshold=segment_confidence,
                text_threshold=segment_confidence,
            )
            obj_mask = merge_masks(obj_masks)
            obj_mask_org = obj_mask
            # obj_mask_org = get_polished_mask(obj_mask_org, 2, mask_config.mask_type)
            # obj_mask = get_polished_mask(obj_mask, mask_config.mask_dilate_kernel, mask_config.mask_type)
            obj_mask = get_polished_mask(obj_mask, mask_config.mask_dilate_kernel, mask_config.mask_type)
            labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
            labeled_img_to_edit = draw_image(
                np.asarray(img_to_edit),
                obj_masks,
                obj_boxes,
                labels
            )

        # Save intermediate img_to_edit and unpolished obj_mask
        labeled_img_to_edit = Image.fromarray(np.uint8(labeled_img_to_edit)).convert("RGB")
        intermediate_images.append(labeled_img_to_edit)
        obj_mask_list.append(obj_mask_org)

        first_iter_obj_mask = obj_mask if i == 0 else first_iter_obj_mask

        # Dream edit
        encode_ratio = encode_ratio_scheduler.step()
        transform = T.Resize((512 // 8, 512 // 8))
        latent_obj_mask = transform(obj_mask[None, ...])[0].to(device)
        _img_edited, img_to_edit_reconstruct, latents_record_list, diffedit_mask = diffedit(
            model=sd_model,
            config=de_config,
            init_image=img_to_edit_path,
            iteration=i,
            src_prompt=src_prompt,
            dst_prompt=dst_prompt,
            encode_ratio=encode_ratio,
            ddim_steps=ddim_config.ddim_steps,
            scale=ddim_config.scale,
            seed=ddim_config.seed,
            sam_mask=latent_obj_mask,
            H=512,
            W=512,
            noise_step=ddim_config.noise_step,
            record_list=latents_record_list,
            use_diffedit=de_config.use_diffedit
        )
        img_edited = Image.fromarray(_img_edited[0])  # type, PIL image

        # Replace the background with the original one
        # previous_masks_merge = merge_masks(torch.stack(obj_mask_list, dim=0))
        # process_mask_difference = subtract_mask(previous_masks_merge, obj_mask_list[-1])
        # if de_config.background_correction_enabled and i <= de_config.background_correction.iteration_number:
        if de_config.background_correction_enabled and i <= de_config.background_correction.iteration_number:
            if not de_config.use_diffedit:
                img_edited = replace_background(
                    src_img,
                    img_edited,
                    obj_class,
                    first_iter_obj_mask,
                    lang_sam_model,
                    lang_sam_config,
                    de_config,
                )

        # Save intermediate mask
        obj_mask_pil = get_mask_pil_image(obj_mask).convert('RGB')  # TODO: why size is not 512?
        obj_mask_pil = obj_mask_pil.resize(img_to_edit.size)
        intermediate_images.append(obj_mask_pil)

        if de_config.use_diffedit:
            diffedit_mask_pil = get_mask_pil_image(diffedit_mask.detach().cpu()).convert('RGB')  # TODO: why size is not 512?
            diffedit_mask_pil = diffedit_mask_pil.resize(img_to_edit.size)
            intermediate_images.append(diffedit_mask_pil)

        # Save intermediate edited image
        tmp_img_name = "tmp_" + config.data.db_folder_name + "_" + config.model.de.task_type
        save_pil_image(pil_image=img_edited, dest_folder="tmp", filename=tmp_img_name, filetype="jpg")
        generated_image_list.append(img_edited)
        img_to_edit_path = os.path.join("tmp", tmp_img_name + ".jpg")
        intermediate_images.append(img_edited)

    # Step2. Postprocessing
    if de_config.inpaint_after_last_iteration:
        # Remove last mask, then postprocessing
        curr_mask = obj_mask_list.pop()
        previous_masks_merge = merge_masks(torch.stack(obj_mask_list, dim=0))
        process_mask = subtract_mask(previous_masks_merge, curr_mask)
        process_mask_pil = get_mask_pil_image(process_mask).convert('RGB')
        intermediate_images.append(process_mask_pil)
        # Inpaint
        inpaint_pipe = select_inpainting_pipeline("sd_inpaint", device)
        img_edited = inpaint_pipe(
            prompt="",
            image=img_edited,
            mask_image=process_mask_pil
        ).images[0]
        intermediate_images.append(img_edited)

    return img_edited, intermediate_images, generated_image_list


def addition(
        src_img_path,
        obj_img_path,  # Useful only for copy-paste
        obj_class,
        lang_sam_model,
        sd_model,
        src_prompt,
        dst_prompt,
        config,
        device,
        labeled_box=None,
):
    de_config = config.model.de
    lang_sam_config = config.model.lang_sam
    segment_confidence = lang_sam_config.segment_confidence

    # Define tmp image path
    tmp_img_name = "tmp_" + config.data.db_folder_name + "_addition"
    tmp_img_path = os.path.join("tmp", tmp_img_name + ".jpg")
    if not os.path.exists(os.path.join("tmp", tmp_img_name)):
        os.makedirs(os.path.join("tmp", tmp_img_name))

    # Step1. Inpaint a sample object into the image
    src_img = Image.open(src_img_path).resize((512, 512))
    intermediate_images = [src_img]
    addition_config = de_config.addition_config

    if addition_config.automate_prompt:
        blip = BLIP_Model()
        caption = get_object_caption(src_img, obj_class, blip, lang_sam_model)
        addition_config_inpaint_prompt = caption
        addition_config_inpaint_phrase = caption
    else:
        addition_config_inpaint_prompt = addition_config.inpaint_prompt
        addition_config_inpaint_phrase = addition_config.inpaint_phrase

    if addition_config.use_copy_paste:
        obj_img = Image.open(obj_img_path).resize((512, 512))
        obj_masks, obj_boxes, phrases, logits = lang_sam_model.predict(
            obj_img,
            obj_class,
            box_threshold=segment_confidence,
            text_threshold=segment_confidence,
        )
        obj_mask = merge_masks(obj_masks)
        mask_config = de_config.mask_config
        # obj_mask = get_polished_mask(obj_mask, mask_config.mask_dilate_kernel, mask_config.mask_type)
        obj_box = bounding_box_merge(obj_boxes)
        obj_mask, inpainted_img = transform_box_mask_paste(labeled_box, obj_box, obj_mask, src_img, obj_img, )
        inpainted_img = Image.fromarray(np.uint8(inpainted_img))
        save_pil_image(pil_image=inpainted_img, dest_folder="tmp", filename=tmp_img_name, filetype="jpg")
        intermediate_images.append(inpainted_img)

    elif addition_config.inpaint_type == "stable_diffusion":
        obj_mask = get_mask_from_bbox(labeled_box).long()
        inpaint_pipe = select_inpainting_pipeline("sd_inpaint", device)
        inpainted_img = inpaint_pipe(
            prompt=addition_config_inpaint_prompt,
            image=src_img_path,
            mask_image=get_mask_pil_image(obj_mask).convert('RGB')
        ).images[0]
        save_pil_image(pil_image=inpainted_img, dest_folder="tmp", filename=tmp_img_name, filetype="jpg")
        intermediate_images.append(inpainted_img)

    elif addition_config.inpaint_type == "gligen":
        # TODO: look into this
        inpaint_pipe = select_inpainting_pipeline("gligen_inpaint", device)
        inpainted_imgs = inpaint_text_gligen(
            pipe=inpaint_pipe,
            prompt=addition_config_inpaint_prompt,
            background_path=src_img_path,
            bounding_box=labeled_box,
            gligen_phrase=addition_config_inpaint_phrase,
            config=config.model.gligen,
        )
        inpainted_imgs = torch.stack(
            [torch.from_numpy(image) for image in inpainted_imgs]
        ).permute(0, 3, 1, 2)
        torchvision.utils.save_image(inpainted_imgs, tmp_img_path, nrow=2, normalize=False)

        # Remove distorted background
        inpainted_img = Image.open(tmp_img_path).resize((512, 512))
        inpainted_img_bg_replaced = inpainted_img
        # I comment the intermediate rectification after gligen as it brings sharp edge
        # inpainted_img_bg_replaced = replace_background(
        #     src_img,
        #     inpainted_img,
        #     obj_class,
        #     None,
        #     lang_sam_model,
        #     lang_sam_config,
        #     de_config,
        # )
        save_pil_image(pil_image=inpainted_img_bg_replaced, dest_folder="tmp", filename=tmp_img_name, filetype="jpg")
        intermediate_images.append(inpainted_img)
        intermediate_images.append(inpainted_img_bg_replaced)
    else:
        raise NotImplementedError

    # Step2. Dream edit the inpainted image
    img_edited, intermediate_images_edited, generated_image_list = iterative_edit(
        src_img_path=src_img_path,
        img_to_edit_path=tmp_img_path,
        obj_class=obj_class,
        lang_sam_model=lang_sam_model,
        sd_model=sd_model,
        src_prompt=src_prompt,
        dst_prompt=dst_prompt,
        config=config,
        device=device,
        labeled_box=labeled_box,
    )
    intermediate_images = intermediate_images + intermediate_images_edited
    intermediate_images = get_concat_pil_images(intermediate_images, direction="h")
    return img_edited, intermediate_images, generated_image_list


def replacement(
        src_img_path,
        obj_img_path,  # Useful only for copy-paste
        obj_class,
        lang_sam_model,
        sd_model,
        src_prompt,
        dst_prompt,
        config,
        device,
):
    # TODO: implement background replacement
    src_img = Image.open(src_img_path).resize((512, 512))
    intermediate_images = [src_img]
    de_config = config.model.de
    lang_sam_config = config.model.lang_sam
    segment_confidence = lang_sam_config.segment_confidence
    replacement_config = de_config.addition_config
    tmp_img_name = "tmp_" + config.data.db_folder_name + "_replacement"
    tmp_img_path = os.path.join("tmp", tmp_img_name + ".jpg")

    if replacement_config.use_copy_paste:
        obj_img = Image.open(obj_img_path).resize((512, 512))
        obj_masks, obj_boxes, phrases, logits = lang_sam_model.predict(
            obj_img,
            obj_class,
            box_threshold=segment_confidence,
            text_threshold=segment_confidence,
        )
        obj_mask = merge_masks(obj_masks)
        obj_box = bounding_box_merge(obj_boxes)

        src_masks, src_boxes, phrases, logits = lang_sam_model.predict(
            src_img,
            obj_class,
            box_threshold=segment_confidence,
            text_threshold=segment_confidence,
        )
        src_mask = merge_masks(src_masks)
        src_box = bounding_box_merge(src_boxes)
        obj_mask, inpainted_img = transform_box_mask_paste(src_box, obj_box, obj_mask, src_img, obj_img, )
        inpainted_img = Image.fromarray(np.uint8(inpainted_img))
        save_pil_image(pil_image=inpainted_img, dest_folder="tmp", filename=tmp_img_name, filetype="jpg")
        intermediate_images.append(inpainted_img)
    else:
        save_pil_image(pil_image=src_img, dest_folder="tmp", filename=tmp_img_name, filetype="jpg")

    img_edited, intermediate_images_edited, generated_image_list = iterative_edit(
        src_img_path=src_img_path,
        img_to_edit_path=tmp_img_path,
        obj_class=obj_class,
        lang_sam_model=lang_sam_model,
        sd_model=sd_model,
        src_prompt=src_prompt,
        dst_prompt=dst_prompt,
        config=config,
        device=device,
    )
    intermediate_images = intermediate_images + intermediate_images_edited
    intermediate_images = get_concat_pil_images(intermediate_images, direction="h")
    return img_edited, intermediate_images, generated_image_list


def dream_edit(
        src_img_path,
        device,
        config,
        obj_img_path=None,
        bbox_file_path=None,
        sd_model=None,
        lang_sam_model=None,
        dino_model=None,
        clip_model=None
):
    src_img_file_name = os.path.basename(src_img_path)
    src_img_file_name_prefix = os.path.splitext(os.path.basename(src_img_path))[0]

    # Create prompt by task type
    task_type = config.model.de.task_type
    special_token = config.model.de.special_token
    class_name = config.data.class_name
    if task_type == "replace" or task_type == "add":
        src_prompt = "photo of a " + class_name
        dst_prompt = "photo of a " + special_token + " " + class_name
    else:
        src_prompt = None
        dst_prompt = None
    print("src_prompt: ", src_prompt)
    print("dst_prompt: ", dst_prompt)

    if task_type == "add":
        with open(bbox_file_path) as json_file:
            bbox_dict = json.load(json_file)
        img_edited, intermediate_images, generated_image_list = addition(
            src_img_path=src_img_path,
            obj_img_path=obj_img_path,
            obj_class=class_name,
            lang_sam_model=lang_sam_model,
            sd_model=sd_model,
            src_prompt=src_prompt,
            dst_prompt=dst_prompt,
            config=config,
            device=device,
            labeled_box=bbox_dict[src_img_file_name],
        )
    elif task_type == "replace":
        img_edited, intermediate_images, generated_image_list = replacement(
            src_img_path=src_img_path,
            obj_img_path=obj_img_path,
            obj_class=class_name,
            lang_sam_model=lang_sam_model,
            sd_model=sd_model,
            src_prompt=src_prompt,
            dst_prompt=dst_prompt,
            config=config,
            device=device,
        )
    else:
        raise NotImplementedError

    save_pil_image(
        pil_image=intermediate_images,
        dest_folder=os.path.join(
            config.experiment_result_path,
            config.data.db_folder_name,
            task_type
        ),
        filename=str(src_img_file_name_prefix) + "_edited",
        filetype="jpg",
    )

    # choose and save the image with the largest dino score
    obj_img_name = config.data.obj_img_file_name
    obj_img_path = os.path.join(
        config.data.db_dataset_path,
        config.data.db_folder_name,
        obj_img_name
    )
    src_img = Image.open(src_img_path).resize((512, 512))
    obj_img = Image.open(obj_img_path).resize((512, 512))
    dino_score_list_subject = evaluate_dino_score_list(obj_img, generated_image_list, device, dino_model)
    dino_score_list_background = evaluate_dino_score_list(src_img, generated_image_list, device, dino_model)
    dino_score_list_average = [round((dino_score_list_subject[x] + dino_score_list_background[x])/2, 3) for x in
                               range(len(dino_score_list_subject))]

    clipi_score_list_subject = evaluate_clipi_score_list(obj_img, generated_image_list, device, clip_model)
    clipi_score_list_background = evaluate_clipi_score_list(src_img, generated_image_list, device, clip_model)
    clipi_score_list_average = [round((clipi_score_list_subject[x] + clipi_score_list_background[x])/2, 3) for x in
                               range(len(clipi_score_list_subject))]


    max_dino_subject = max(dino_score_list_subject)
    max_dino_background = max(dino_score_list_background)
    max_dino_avg = max(dino_score_list_average)

    max_clipi_subject = max(clipi_score_list_subject)
    max_clipi_background = max(clipi_score_list_background)
    max_clipi_avg = max(clipi_score_list_average)

    max_index_dino_subject = dino_score_list_subject.index(max_dino_subject)
    max_index_dino_background = dino_score_list_background.index(max_dino_background)
    max_index_dino_avg = dino_score_list_average.index(max_dino_avg)

    max_index_clipi_subject = clipi_score_list_subject.index(max_clipi_subject)
    max_index_clipi_background = clipi_score_list_background.index(max_clipi_background)
    max_index_clipi_avg = clipi_score_list_average.index(max_clipi_avg)

    print("The best result of dino-subject is at {} th iteration with dino score {}".format(max_index_dino_subject + 1,
                                                                                            max_dino_subject))
    print("The best result of dino-background is at {} th iteration with dino score {}".format(max_index_dino_background + 1,
                                                                                            max_dino_background))
    print("The best result of dino-avg is at {} th iteration with dino score {}".format(
        max_index_dino_avg + 1,
        max_dino_avg))


    print("The best result of clipi-subject is at {} th iteration with clipi score {}".format(max_index_clipi_subject + 1,
                                                                                            max_clipi_subject))
    print("The best result of clipi-background is at {} th iteration with clipi score {}".format(max_index_clipi_background + 1,
                                                                                            max_clipi_background))
    print("The best result of clipi-avg is at {} th iteration with clipi score {}".format(
        max_index_clipi_avg + 1,
        max_clipi_avg))

    # ====== Max: Haven't saved clip-i score
    final_result = [src_img, generated_image_list[max_index_dino_avg]]
    final_result = get_concat_pil_images(final_result, direction="h")
    save_pil_image(
        pil_image=final_result,
        dest_folder=os.path.join(
            config.experiment_result_path,
            config.data.db_folder_name,
            task_type
        ),
        filename=str(src_img_file_name_prefix) + "_edited_final",
        filetype="jpg",
    )
    return max_dino_background, max_dino_subject, max_dino_avg, max_clipi_background, max_clipi_subject, max_clipi_avg, intermediate_images


def edit_all_images_from_class(config):
    # Load model
    sd_ckpt_path = config.model.sd.ckpt_path
    sd_conf = OmegaConf.load(config.model.sd.conf_path)
    sd_model = load_model_from_config(sd_conf, sd_ckpt_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sd_model = sd_model.to(device)
    fidelity = VITs16(device)
    clip_model = CLIP(device)
    langSAM = LangSAM()

    # load all base images belonging to the class
    task_type = config.model.de.task_type
    class_name = config.data.class_name
    folder_name = config.data.db_folder_name
    obj_img_file_name = config.data.obj_img_file_name
    src_img_data_folder_path = config.data.src_img_data_folder_path
    all_files_path = os.path.join(src_img_data_folder_path, config.benchmark_folder_name, "*")
    print("Looking for images in path: ", all_files_path)
    all_files = sorted(glob.glob(all_files_path))

    if task_type == "edit":
        all_files = [img_name for img_name in all_files if img_name.endswith(obj_img_file_name)] * 3

    final_score_avg_list = []
    final_score_subject_list = []
    final_score_background_list = []

    final_score_avg_clipi_list = []
    final_score_subject_clipi_list = []
    final_score_background_clipi_list = []

    all_images_to_save = []

    print("Found files: ", all_files)

    for file_path in all_files:
        if file_path.endswith(".jpg"):
            print("Processing img:", file_path)
            bbox_file_path = os.path.join(
                config.data.src_img_data_folder_path,
                config.benchmark_folder_name,
                config.data.bbox_file_name,
            )
            obj_img_path = os.path.join(
                config.data.db_dataset_path,
                config.data.db_folder_name,
                config.data.obj_img_file_name,
            )
            max_dino_background, max_dino_subject, max_dino_avg, max_clipi_background, max_clipi_subject, max_clipi_avg, intermediate_images = dream_edit(
                src_img_path=file_path,
                device=device,
                config=config,
                obj_img_path=obj_img_path,
                bbox_file_path=bbox_file_path,
                sd_model=sd_model,
                lang_sam_model=langSAM,
                dino_model=fidelity,
                clip_model=clip_model
            )
            final_score_avg_list.append(max_dino_avg)
            final_score_subject_list.append(max_dino_subject)
            final_score_background_list.append(max_dino_background)
            final_score_avg_clipi_list.append(max_clipi_avg)
            final_score_subject_clipi_list.append(max_clipi_subject)
            final_score_background_clipi_list.append(max_clipi_background)
            all_images_to_save.append(intermediate_images)

    print("The average dino-subject score is: {}".format(sum(final_score_subject_list) / len(final_score_subject_list)))
    print("The average dino-background score is: {}".format(sum(final_score_background_list) / len(final_score_background_list)))
    print("The average dino-avg score is: {}".format(sum(final_score_avg_list) / len(final_score_avg_list)))
    print("The average clipi-subject score is: {}".format(sum(final_score_subject_clipi_list) / len(final_score_subject_clipi_list)))
    print("The average clipi-background score is: {}".format(
        sum(final_score_background_clipi_list) / len(final_score_background_clipi_list)))
    print("The average clipi-avg score is: {}".format(sum(final_score_avg_clipi_list) / len(final_score_avg_clipi_list)))
    all_images_concat = get_concat_pil_images(all_images_to_save, direction="v")
    save_pil_image(
        pil_image=all_images_concat,
        dest_folder=os.path.join(
            config.experiment_result_path,
            config.data.db_folder_name,
            task_type
        ),
        filename="result_all",
        filetype="jpg",
    )


def edit_single_img_from_class(config):
    # Load model
    sd_ckpt_path = config.model.sd.ckpt_path
    sd_conf = OmegaConf.load(config.model.sd.conf_path)
    sd_model = load_model_from_config(sd_conf, sd_ckpt_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sd_model = sd_model.to(device)
    fidelity = VITs16(device)
    clip_model = CLIP(device)
    langSAM = LangSAM()
    src_img_path = os.path.join(
        config.data.src_img_data_folder_path,
        config.benchmark_folder_name,
        config.data.src_img_file_name,
    )
    bbox_file_path = os.path.join(
        config.data.src_img_data_folder_path,
        config.benchmark_folder_name,
        config.data.bbox_file_name,
    )
    obj_img_path = os.path.join(
        config.data.db_dataset_path,
        config.data.db_folder_name,
        config.data.obj_img_file_name,
    )
    max_dino_background, max_dino_subject, max_dino_avg, intermediate_images = dream_edit(
        src_img_path=src_img_path,
        device=device,
        config=config,
        obj_img_path=obj_img_path,
        bbox_file_path=bbox_file_path,
        sd_model=sd_model,
        lang_sam_model=langSAM,
        dino_model=fidelity,
        clip_model=clip_model
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--edit_single_img_from_class",
        action="store_true",
        help="edit a single image from a class",
    )
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config)
    if opt.edit_single_img_from_class:
        edit_single_img_from_class(config)
    else:
        edit_all_images_from_class(config)
