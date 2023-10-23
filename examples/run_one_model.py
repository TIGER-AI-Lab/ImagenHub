import imagen_hub
from imagen_hub.utils import save_pil_image

print(imagen_hub.__version__)
model = imagen_hub.load("SDXL")
image = model.infer_one_image(prompt="people reading pictures in a museum, watercolor", seed=42)
save_pil_image(image, ".", "output1.jpg")
