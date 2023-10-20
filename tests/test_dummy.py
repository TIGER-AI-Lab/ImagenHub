import imagen_hub

print(imagen_hub.__version__)
model = imagen_hub.load("SDXL")
image = model.infer_one_image(prompt="people reading pictures in a museum, watercolor", seed=42)
assert image.size == (1024, 1024)
