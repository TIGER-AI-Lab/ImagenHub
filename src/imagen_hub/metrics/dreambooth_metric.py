import numpy as np
import torch
from PIL import Image

from imagen_hub.miscmodels.dino_vit import VITs16
from imagen_hub.miscmodels.clip_vit import CLIP

"""
DreamBooth metrics

* DINO
* CLIP-I
* CLIP-T
"""
class MetricDINO():
    def __init__(self, device="cuda") -> None:
        self.model = VITs16(device)
        self.device = device

    def evaluate(self, real_image: Image.Image, generated_image: Image.Image):
        """
        return distance
        """
        return evaluate_dino_score(real_image, generated_image, self.model, self.device)

class MetricCLIP_I():
    def __init__(self, device="cuda") -> None:
        self.model = CLIP(device)
        self.device = device

    def evaluate(self, real_image: Image.Image, generated_image: Image.Image):
        """
        return distance
        """
        return evaluate_clipi_score(real_image, generated_image, self.model, self.device)

class MetricCLIP_T():
    def __init__(self, device="cuda") -> None:
        self.model = CLIP(device)
        self.device = device

    def evaluate(self, generated_image: Image.Image, text: str):
        """
        return distance
        """
        return evaluate_clipt_score(generated_image, text, self.model, self.device)

def compute_cosine_distance(image_features, image_features2):
    # normalized features
    image_features = image_features / np.linalg.norm(np.float32(image_features), ord=2)
    image_features2 = image_features2 / np.linalg.norm(np.float32(image_features2), ord=2)
    return np.dot(image_features, image_features2)

def compute_l2_distance(image_features, image_features2):
    return np.linalg.norm(np.float32(image_features - image_features2))

def evaluate_dino_score(real_image, generated_image, fidelity_model, device):
    #tensor_image_1 = torch.from_numpy(np.asarray(real_image)).permute(2, 0, 1).unsqueeze(0)
    #tensor_image_2 = torch.from_numpy(np.asarray(generated_image)).permute(2, 0, 1).unsqueeze(0)
    preprocess = fidelity_model.get_transform()
    tensor_image_1 = preprocess(real_image).unsqueeze(0)
    tensor_image_2 = preprocess(generated_image).unsqueeze(0)
    emb_1 = fidelity_model.get_embeddings(tensor_image_1.float().to(device))
    emb_2 = fidelity_model.get_embeddings(tensor_image_2.float().to(device))
    assert emb_1.shape == emb_2.shape
    score = compute_cosine_distance(emb_1.detach().cpu().numpy(), emb_2.detach().permute(1, 0).cpu().numpy())
    return score[0][0]

def evaluate_dino_score_list(real_image, generated_image_list, fidelity_model, device):
    score_list = []
    total = len(generated_image_list)
    for i in range(total):
        score = evaluate_dino_score(real_image, generated_image_list[i], device, fidelity_model)
        score_list.append(score)
    # max_score = max(score_list)
    # max_index = score_list.index(max_score)
    # print("The best result is at {} th iteration with dino score {}".format(max_index + 1, max_score))
    # return max_score, max_index
    return score_list

def evaluate_clipi_score(real_image, generated_image, clip_model, device):
    preprocess = clip_model.get_transform()
    tensor_image_1 = preprocess(real_image).unsqueeze(0)
    tensor_image_2 = preprocess(generated_image).unsqueeze(0)
    emb_1 = clip_model.encode_image(tensor_image_1.float().to(device))
    emb_2 = clip_model.encode_image(tensor_image_2.float().to(device))
    assert emb_1.shape == emb_2.shape
    score = compute_cosine_distance(emb_1.detach().cpu().numpy(), emb_2.detach().permute(1, 0).cpu().numpy())
    return score[0][0]

def evaluate_clipi_score_list(real_image, generated_image_list, clip_model, device):
    score_list = []
    total = len(generated_image_list)
    for i in range(total):
        score = evaluate_clipi_score(real_image, generated_image_list[i], device, clip_model)
        score_list.append(score)
    # max_score = max(score_list)
    # max_index = score_list.index(max_score)
    # print("The best result is at {} th iteration with dino score {}".format(max_index + 1, max_score))
    # return max_score, max_index
    return score_list

def evaluate_clipt_score(generated_image, text, clip_model, device):
    preprocess = clip_model.get_transform()
    tensor_image_1 = preprocess(generated_image).unsqueeze(0)
    emb_1 = clip_model.encode_image(tensor_image_1.float().to(device))
    emb_2 = clip_model.encode_text(text)
    assert emb_1.shape == emb_2.shape
    score = compute_cosine_distance(emb_1.detach().cpu().numpy(), emb_2.detach().permute(1, 0).cpu().numpy())
    return score[0][0]