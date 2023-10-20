from imagen_hub.miscmodels import LangSAM, CLIP, VITs16
from imagen_hub.depend.lang_sam.lang_sam import draw_image
from imagen_hub.utils.configs import get_SD_conf
from imagen_hub.utils.save_image_helper import get_concat_pil_images, get_mask_pil_image, save_pil_image
from imagen_hub.metrics.dreambooth_metric import evaluate_dino_score_list, evaluate_clipi_score_list

from .dreamedit_src.utils.mask_helper import merge_masks, subtract_mask, get_polished_mask, mask_dilation, mask_closing, mask_closing_half, transform_box_mask, transform_box_mask_paste, resize_box_from_middle, resize_box_from_bottom, bounding_box_merge, get_mask_from_bbox
from .dreamedit_src.generate import diffedit, load_model_from_config
from .dreamedit_src.iterate_generate import EncodeRatioScheduler
from .dreamedit_src.pipelines.inpainting_pipelines import select_inpainting_pipeline, inpaint_text_gligen
from .dreamedit_src.pipelines.extract_object_pipeline import get_object_caption

import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

from omegaconf import OmegaConf
import os

class DreamEditPipeline():
    def __init__(self, device="cuda") -> None:
        self.device=device
        self.lang_sam_model=LangSAM()

        # Waiting to init
        self.config=None
        self.sd_model=None
        self.dino_model=None
        self.clip_model=None

        self.class_name=None
        self.special_token=None
        self.src_prompt=None
        self.dst_prompt=None

    def set_default_config(self, type="replace"):
        """
        loading default config from dreamedit and override it afterwards
        hardcoded yaml file
        """
        if type == "add":
            # Edit: copied from experiments/add_dog8_config_01.yaml
            self.config = OmegaConf.create({'base_path': '${oc.env:HOME}', 'experiment_name': 'add_dog8', 'class_obj_name': 'dog', 'class_folder_name': 'dog8', 'benchmark_folder_name': 'dog', 'token_of_class': 'wie', 'ckpt_base_folder': 'dog82023-04-17T00-58-03_dog8_april/', 'dream_edit_prompt': 'a grey and white border collie dog', 'config_name': 'add_${class_folder_name}_config_01', 'experiment_result_path': '/home/data/dream_edit_project/results/${experiment_name}/${config_name}/', 'data': {'src_img_data_folder_path': '/home/data/dream_edit_project/benchmark/background_images_refine/', 'class_name': '${class_obj_name}', 'bbox_file_name': 'bbox.json', 'src_img_file_name': 'found0.jpg', 'db_dataset_path': '/home/data/dream_edit_project/benchmark/cvpr_dataset/', 'db_folder_name': '${class_folder_name}', 'obj_img_file_name': '00.jpg'}, 'model': {'gligen': {'gligen_scheduled_sampling_beta': 1, 'num_inference_steps': 100}, 'lang_sam': {'segment_confidence': 0.1}, 'sd': {'conf_path': 'configs/stable-diffusion/v1-inference.yaml', 'ckpt_prefix': '/home/data/dream_edit_project/model_weights/', 'ckpt': '${ckpt_base_folder}', 'ckpt_suffix': 'checkpoints/last.ckpt', 'ckpt_path': '${model.sd.ckpt_prefix}${model.sd.ckpt}${model.sd.ckpt_suffix}'}, 'de': {'task_type': 'add', 'special_token': '${token_of_class}', 'bounding_box': 'bbox.json', 'inpaint_after_last_iteration': False, 'postprocessing_type': 'sd_inpaint', 'use_diffedit': False, 'addition_config': {'use_copy_paste': False, 'inpaint_type': 'gligen', 'automate_prompt': False, 'inpaint_prompt': 'photo of ${dream_edit_prompt}', 'inpaint_phrase': '${dream_edit_prompt}'}, 'mask_config': {'mask_dilate_kernel': 20, 'mask_type': 'dilation', 'use_bbox_mask_for_first_iteration': True, 'use_bbox_mask_for_all_iterations': False}, 'ddim': {'seed': 42, 'scale': 5.5, 'ddim_steps': 40, 'noise_step': 0, 'iteration_number': 7, 'encode_ratio_schedule': {'decay_type': 'manual', 'start_ratio': 0.8, 'end_ratio': 0.3, 'manual_ratio_list': [0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3]}}, 'background_correction_enabled': True, 'background_correction': {'iteration_number': 7, 'use_latents_record': False, 'use_background_from_original_image': True, 'use_obj_mask_from_first_iteration': False}}}})
        elif type == "replace":
            # Add: copied from experiments/replace_dog8_config_01.yaml
            self.config = OmegaConf.create({'base_path': '${oc.env:HOME}', 'experiment_name': 'replace_dog8', 'class_obj_name': 'dog', 'class_folder_name': 'dog8', 'benchmark_folder_name': 'dog', 'token_of_class': 'wie', 'ckpt_base_folder': 'dog82023-04-17T00-58-03_dog8_april/', 'dream_edit_prompt': 'a grey and white border collie dog', 'config_name': 'replace_${class_folder_name}_config_01', 'experiment_result_path': '/home/data/dream_edit_project/results/${experiment_name}/${config_name}/', 'data': {'src_img_data_folder_path': '/home/data/dream_edit_project/benchmark/ref_images/', 'class_name': '${class_obj_name}', 'bbox_file_name': 'bbox.json', 'src_img_file_name': 'found0.jpg', 'db_dataset_path': '/home/data/dream_edit_project/benchmark/cvpr_dataset/', 'db_folder_name': '${class_folder_name}', 'obj_img_file_name': '00.jpg'}, 'model': {'gligen': {'gligen_scheduled_sampling_beta': 1, 'num_inference_steps': 100}, 'lang_sam': {'segment_confidence': 0.1}, 'sd': {'conf_path': 'configs/stable-diffusion/v1-inference.yaml', 'ckpt_prefix': '/home/data/dream_edit_project/model_weights/', 'ckpt': '${ckpt_base_folder}', 'ckpt_suffix': 'checkpoints/last.ckpt', 'ckpt_path': '${model.sd.ckpt_prefix}${model.sd.ckpt}${model.sd.ckpt_suffix}'}, 'de': {'task_type': 'replace', 'special_token': '${token_of_class}', 'bounding_box': 'bbox.json', 'inpaint_after_last_iteration': False, 'postprocessing_type': 'sd_inpaint', 'use_diffedit': False, 'addition_config': {'use_copy_paste': False, 'inpaint_type': 'gligen', 'automate_prompt': False, 'inpaint_prompt': 'photo of ${dream_edit_prompt}', 'inpaint_phrase': '${dream_edit_prompt}'}, 'mask_config': {'mask_dilate_kernel': 20, 'mask_type': 'dilation', 'use_bbox_mask_for_first_iteration': False, 'use_bbox_mask_for_all_iterations': False}, 'ddim': {'seed': 42, 'scale': 5.5, 'ddim_steps': 40, 'noise_step': 0, 'iteration_number': 5, 'encode_ratio_schedule': {'decay_type': 'linear', 'start_ratio': 0.6, 'end_ratio': 0.3, 'manual_ratio_list': [0.5, 0.4, 0.4, 0.4, 0.3]}}, 'background_correction_enabled': True, 'background_correction': {'iteration_number': 3, 'use_latents_record': False, 'use_background_from_original_image': True, 'use_obj_mask_from_first_iteration': True}}}})
        else:
            raise NotImplementedError

    def set_seed(self, seed):
        """
        override seed value in config.
        """
        if self.config != None:
            ddim_config = self.config.model.de.ddim
            ddim_config.seed = seed
        else:
            raise AttributeError("Use `set_default_config()` to init the config first")

    def set_sd_model(self, ckpt_path):
        sd_ckpt_path = ckpt_path
        sd_conf = get_SD_conf()
        self.sd_model = load_model_from_config(sd_conf, sd_ckpt_path).to(self.device)

    def set_ddim_config(self, ddim_steps=40, scale=5.5, noise_step=0, iteration_number=5):
        if self.config != None:
            ddim_config = self.config.model.de.ddim
            ddim_config.scale = scale
            ddim_config.ddim_steps = ddim_steps
            ddim_config.noise_step = noise_step
            ddim_config.iteration_number= iteration_number
        else:
            raise AttributeError("Use `set_default_config()` to init the config first")

    def set_scoring_model(self, dino_only=True):
        fidelity = VITs16(self.device)
        self.dino_model = fidelity
        if not dino_only:
            clip_model = CLIP(self.device)
            self.clip_model = clip_model

    def set_subject(self, class_name, special_token):
        """
        class_name : str the object class
        special_token
        """
        self.class_name = class_name
        self.special_token = special_token

    def set_prompts(self, src_prompt, dst_prompt):
        """
        Setting custom prompts
        """
        self.src_prompt=src_prompt
        self.dst_prompt=dst_prompt

    def set_default_prompts(self):
        """
            self.src_prompt = "photo of a " + self.class_name
            self.dst_prompt = "photo of a " + self.special_token + " " + self.class_name
        """
        if self.class_name is not None and self.special_token is not None:
            self.src_prompt = "photo of a " + self.class_name
            self.dst_prompt = "photo of a " + self.special_token + " " + self.class_name
        else:
            raise AttributeError("self.class_name and self.special_token should not be none")

    def get_default_prompts(self):
        if self.class_name is not None and self.special_token is not None:
            src_prompt = "photo of a " + self.class_name
            dst_prompt = "photo of a " + self.special_token + " " + self.class_name
            return src_prompt, dst_prompt
        else:
            raise AttributeError("self.class_name and self.special_token should not be none")

    def infer(
            self,
            src_image,
            obj_image=None,
    ):
        # Create prompt by task type
        task_type = self.config.model.de.task_type
        if task_type == "add":
            raise NotImplementedError # we implement it later
        elif task_type == "replace":
            img_edited, intermediate_images, generated_image_list = self.replacement(
                src_image=src_image,
            )
        else:
            raise NotImplementedError

        if(obj_image is not None):
            self.set_scoring_model(dino_only=True) # Awake Dino Metric
            # choose and save the image with the largest dino score
            dino_score_list_subject = evaluate_dino_score_list(obj_image, generated_image_list, self.device, self.dino_model)
            dino_score_list_background = evaluate_dino_score_list(src_image, generated_image_list, self.device, self.dino_model)
            dino_score_list_average = [round((dino_score_list_subject[x] + dino_score_list_background[x])/2, 3) for x in
                                    range(len(dino_score_list_subject))]
            max_dino_avg = max(dino_score_list_average)

            max_index_dino_avg = dino_score_list_average.index(max_dino_avg)
            final_result = generated_image_list[max_index_dino_avg]
            return final_result
        else:
            return img_edited

    def addition(
            self,
            src_image,
            labeled_box=None,
    ):
        raise NotImplementedError()

    def replacement(
            self,
            src_image,
    ):
        intermediate_images = [src_image]

        img_edited, intermediate_images_edited, generated_image_list = self.iterative_edit(
            src_image=src_image
        )
        intermediate_images = intermediate_images + intermediate_images_edited
        intermediate_images = get_concat_pil_images(intermediate_images, direction="h")
        return img_edited, intermediate_images, generated_image_list

    def iterative_edit(
            self,
            src_image,
            labeled_box=None,
    ):

        src_image = src_image.resize((512, 512))
        img_to_edit = src_image

        obj_class = self.class_name

        de_config = self.config.model.de
        lang_sam_config = self.config.model.lang_sam
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
            raise NotImplementedError # we implement it later
        else:
            pbar = tqdm(range(ddim_config.iteration_number))
        for i in pbar:
            if de_config.use_diffedit:
                raise NotImplementedError # we implement it later
            else:
                pbar.set_description("Dream edit iteration %s" % i)

            # Get object mask
            mask_config = de_config.mask_config
            if (mask_config.use_bbox_mask_for_first_iteration and i == 0) or mask_config.use_bbox_mask_for_all_iterations:
                obj_mask = get_mask_from_bbox(labeled_box).long()

                obj_masks, obj_boxes, phrases, logits = self.lang_sam_model.predict_adaptive(
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
                obj_masks, obj_boxes, phrases, logits = self.lang_sam_model.predict_adaptive(
                    img_to_edit,
                    obj_class,
                    box_threshold=segment_confidence,
                    text_threshold=segment_confidence,
                )
                obj_mask = merge_masks(obj_masks)
                obj_mask_org = obj_mask
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
            latent_obj_mask = transform(obj_mask[None, ...])[0].to(self.device)
            _img_edited, img_to_edit_reconstruct, latents_record_list, diffedit_mask = diffedit(
                model=self.sd_model,
                config=de_config,
                init_image=img_to_edit,
                iteration=i,
                src_prompt=self.src_prompt,
                dst_prompt=self.dst_prompt,
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
            if de_config.background_correction_enabled and i <= de_config.background_correction.iteration_number:
                if not de_config.use_diffedit:
                    img_edited = self.replace_background(
                        src_image,
                        img_edited,
                        obj_class,
                        first_iter_obj_mask
                    )

            # Save intermediate mask
            obj_mask_pil = get_mask_pil_image(obj_mask).convert('RGB')  # TODO: why size is not 512?
            obj_mask_pil = obj_mask_pil.resize(img_to_edit.size)
            intermediate_images.append(obj_mask_pil)

            if de_config.use_diffedit:
                diffedit_mask_pil = get_mask_pil_image(diffedit_mask.detach().cpu()).convert('RGB')  # TODO: why size is not 512?
                diffedit_mask_pil = diffedit_mask_pil.resize(img_to_edit.size)
                intermediate_images.append(diffedit_mask_pil)

            generated_image_list.append(img_edited)
            intermediate_images.append(img_edited)

            #Update edited image to current image
            img_to_edit = img_edited

        # Step2. Postprocessing (Unused)
        if de_config.inpaint_after_last_iteration:
            # Remove last mask, then postprocessing
            curr_mask = obj_mask_list.pop()
            previous_masks_merge = merge_masks(torch.stack(obj_mask_list, dim=0))
            process_mask = subtract_mask(previous_masks_merge, curr_mask)
            process_mask_pil = get_mask_pil_image(process_mask).convert('RGB')
            intermediate_images.append(process_mask_pil)
            # Inpaint
            inpaint_pipe = select_inpainting_pipeline("sd_inpaint", self.device)
            img_edited = inpaint_pipe(
                prompt="",
                image=img_edited,
                mask_image=process_mask_pil
            ).images[0]
            intermediate_images.append(img_edited)

        return img_edited, intermediate_images, generated_image_list

    def replace_background(
            self,
            src_img,
            img_to_edit,
            obj_class,
            first_iter_obj_mask,
            difference = False
    ):
        assert src_img.size == img_to_edit.size
        de_config = self.config.model.de
        lang_sam_config = self.config.model.lang_sam
        bg_config = de_config.background_correction
        if bg_config.use_obj_mask_from_first_iteration and first_iter_obj_mask is not None:
            obj_mask = first_iter_obj_mask
            if difference:
                masks, boxes, phrases, logits = self.lang_sam_model.predict_adaptive(
                    img_to_edit,
                    obj_class,
                    box_threshold=lang_sam_config.segment_confidence,
                    text_threshold=lang_sam_config.segment_confidence,
                )
                obj_mask = merge_masks(masks).long()
                obj_mask = subtract_mask(first_iter_obj_mask, obj_mask).bool() == False
                obj_mask = obj_mask.long()
        else:
            masks, boxes, phrases, logits = self.lang_sam_model.predict_adaptive(
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
