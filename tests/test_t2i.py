import torch

# Directly run `python -m pytest` or
# Directly run `python -m pytest -v -s --disable-warnings` for Debugging

# To test single function:
# pytest tests/test_t2i.py::test_function_name

dummy_prompts = [
    "a teddy bear walking on the street, 2k, high quality",
    "a panda taking a selfie, 2k, high quality",
    "a polar bear playing drum kit in NYC Times Square, 4k, high resolution",
    "jungle river at sunset, ultra quality",
    "a shark swimming in clear Carribean ocean, 2k, high quality",
    "a Corgi walking in the park at sunrise, oil painting style",
]


def test_SD():
    from imagen_hub.infermodels.sd import SD

    model = SD()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_OpenJourney():
    from imagen_hub.infermodels.sd import OpenJourney

    model = OpenJourney()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_LCM():
    from imagen_hub.infermodels.sd import LCM

    model = LCM()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_PlayGroundV2():
    from imagen_hub.infermodels.sd import PlayGroundV2

    model = PlayGroundV2()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_StableCascade():
    from imagen_hub.infermodels.sd import StableCascade

    model = StableCascade()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_SDXL():
    from imagen_hub.infermodels.sdxl import SDXL

    model = SDXL()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_SDXLTurbo():
    from imagen_hub.infermodels.sdxl import SDXLTurbo

    model = SDXLTurbo()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_SSD():
    from imagen_hub.infermodels.sdxl import SSD

    model = SSD()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_SDXLLightning():
    from imagen_hub.infermodels.sdxl import SDXLLightning

    model = SDXLLightning()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_DeepFloydIF():
    from imagen_hub.infermodels.deepfloydif import DeepFloydIF

    model = DeepFloydIF()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_DALLE2():
    from imagen_hub.infermodels.dalle import DALLE2

    model = DALLE2()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_DALLE3():
    from imagen_hub.infermodels.dalle import DALLE3

    model = DALLE3()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_StableUnCLIP():
    from imagen_hub.infermodels.dalle import StableUnCLIP

    model = StableUnCLIP()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_UniDiffuser():
    from imagen_hub.infermodels.unidiffuser import UniDiffuser

    model = UniDiffuser()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_Kandinsky():
    from imagen_hub.infermodels.kandinsky import Kandinsky

    model = Kandinsky()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_PixArtAlpha():
    from imagen_hub.infermodels.pixart_alpha import PixArtAlpha

    model = PixArtAlpha()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_Wuerstchen():
    from imagen_hub.infermodels.wuerstchen import Wuerstchen

    model = Wuerstchen()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)


def test_CosXL():
    from imagen_hub.infermodels.cosxl import CosXL

    model = CosXL()
    assert model is not None
    out_image = model.infer_one_image(dummy_prompts[0])
    assert out_image is not None
    # check if out_image is a PIL.Image.Image or not
    assert isinstance(out_image, torch.Tensor)
    print(out_image.shape)
