from imagen_hub.metrics import MetricLPIPS
from imagen_hub.utils import load_image, save_pil_image, get_concat_pil_images

def evaluate_one(model, real_image, generated_image):
  score = model.evaluate(real_image, generated_image)
  print("====> Score : ", score)

image_I = load_image("https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_102724_1.jpg")
image_O = load_image("https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_102724_1.jpg")
show_image = get_concat_pil_images([image_I, image_O], 'h')

model = MetricLPIPS()
evaluate_one(model, image_I, image_O)

# You can perform multiple images in the similar manner
def evaluate_all(model, list_real_images, list_generated_images):
  score = [model.evaluate(x, y) for (x,y) in zip(list_real_images, list_generated_images)]
  print("====> Avg Score: ", sum(score) / len(score))

save_pil_image(show_image, ".", "output2.jpg")