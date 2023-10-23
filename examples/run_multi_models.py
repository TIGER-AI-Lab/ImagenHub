import imagen_hub
from imagen_hub.loader.infer_dummy import load_text_guided_ig_data
from imagen_hub.utils import save_pil_image, get_concat_pil_images

dummy_data = load_text_guided_ig_data(get_one_data=True)
print(dummy_data)
prompt = dummy_data['prompt']
model_list = ["SD", "SDXL"]
image_list = []
for model_name in model_list:
    model = imagen_hub.load(model_name)
    output = model.infer_one_image(prompt=prompt, seed=42).resize((512,512))
    image_list.append(output) 

show_image = get_concat_pil_images(image_list)
save_pil_image(show_image, ".", "output3.jpg")