import pytest
import torch
import PIL

if not torch.cuda.is_available():
    pytest.skip("no GPU available.", allow_module_level=True)

from nnsight.modeling.diffusion import DiffusionModel

@pytest.fixture(scope="module")
def tiny_sd():
    return DiffusionModel("segmind/tiny-sd", torch_dtype=torch.float16, dispatch=True).to("cuda")

@pytest.fixture(scope="module")
def cat_prompt():
    return "A brown and white cat staring off with pretty green eyes"

@pytest.fixture(scope="module")
def panda_prompt():
    return "A red panda eating a bamboo"

@pytest.fixture(scope="module")
def birthday_cake_prompt():
    return "A birthday cake with candles"

@pytest.fixture(scope="module")
def wave_prompt():
    return "The great wave off Kanagawa"


@torch.no_grad()
def test_generation(tiny_sd, cat_prompt):
    num_images_per_prompt = 3

    images = tiny_sd.generate(cat_prompt, 
                              num_inference_steps=20, 
                              num_images_per_prompt=num_images_per_prompt,
                              seed=423, 
                              trace=False).images

    assert len(images) == num_images_per_prompt
    assert all([type(img) == PIL.Image.Image for img in images])
    assert all(images[i] != images[j] for i in range(num_images_per_prompt) for j in range(i + 1, num_images_per_prompt))


@torch.no_grad()
def test_text_encoder_batching(
    tiny_sd, 
    cat_prompt,
    panda_prompt,
    birthday_cake_prompt,
    wave_prompt
):

    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):
            encoder_out_1 = tiny_sd.text_encoder.text_model.encoder.layers[-1].output[0].save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            encoder_out_2 = tiny_sd.text_encoder.text_model.encoder.layers[-1].output[0].save()

        with tracer.invoke(wave_prompt):
            encoder_out_3 = tiny_sd.text_encoder.text_model.encoder.layers[-1].output[0].save()

        with tracer.invoke():
            encoder_out_all = tiny_sd.text_encoder.text_model.encoder.layers[-1].output[0].save()

    assert encoder_out_all.shape[0] == encoder_out_1.shape[0] + encoder_out_2.shape[0] + encoder_out_3.shape[0]
    assert encoder_out_1.shape[0] == 1
    assert encoder_out_2.shape[0] == 2
    assert encoder_out_3.shape[0] == 1

    assert torch.all(encoder_out_1 == encoder_out_all[0:1])
    assert torch.all(encoder_out_2 == encoder_out_all[1:3])
    assert torch.all(encoder_out_3 == encoder_out_all[3:])

@torch.no_grad()
def test_unet_batching(
    tiny_sd, 
    cat_prompt,
    panda_prompt,
    birthday_cake_prompt,
    wave_prompt
):
    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        guidance_scale=1.0,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):
            unet_out_1 = tiny_sd.unet.output[0].save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            unet_out_2 = tiny_sd.unet.output[0].save()

        with tracer.invoke(wave_prompt):
            unet_out_3 = tiny_sd.unet.output[0].save()

        with tracer.invoke():
            unet_out_all = tiny_sd.unet.output[0].save()

    assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
    assert unet_out_1.shape[0] == 3
    assert unet_out_2.shape[0] == 6
    assert unet_out_3.shape[0] == 3

    assert torch.all(unet_out_1 == unet_out_all[0:3])
    assert torch.all(unet_out_2 == unet_out_all[3:9])
    assert torch.all(unet_out_3 == unet_out_all[9:])

@torch.no_grad()
def test_unet_batching_with_guidance(
    tiny_sd, 
    cat_prompt, 
    panda_prompt, 
    birthday_cake_prompt, 
    wave_prompt
):

    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        guidance_scale=7.5,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):
            unet_out_1 = tiny_sd.unet.output[0].save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            unet_out_2 = tiny_sd.unet.output[0].save()

        with tracer.invoke(wave_prompt):
            unet_out_3 = tiny_sd.unet.output[0].save()

        with tracer.invoke():
            unet_out_all = tiny_sd.unet.output[0].save()

    assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
    assert unet_out_1.shape[0] == 6
    assert unet_out_2.shape[0] == 12
    assert unet_out_3.shape[0] == 6

    assert torch.all(unet_out_1 == torch.cat([unet_out_all[0:3], unet_out_all[12:15]], dim=0))
    assert torch.all(unet_out_2 == torch.cat([unet_out_all[3:9], unet_out_all[15:21]], dim=0))
    assert torch.all(unet_out_3 == torch.cat([unet_out_all[9:12], unet_out_all[21:24]], dim=0))

@torch.no_grad()
def test_vae_batching(
    tiny_sd, 
    cat_prompt,
    panda_prompt,
    birthday_cake_prompt,
    wave_prompt
):
    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        guidance_scale=1.0,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):
            vae_decoder_out_1 = tiny_sd.vae.decoder.output.save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            vae_decoder_out_2 = tiny_sd.vae.decoder.output.save()

        with tracer.invoke(wave_prompt):
            vae_decoder_out_3 = tiny_sd.vae.decoder.output.save()

        with tracer.invoke():
            vae_decoder_out_all = tiny_sd.vae.decoder.output.save()

    assert vae_decoder_out_all.shape[0] == vae_decoder_out_1.shape[0] + vae_decoder_out_2.shape[0] + vae_decoder_out_3.shape[0]
    assert vae_decoder_out_1.shape[0] == 3
    assert vae_decoder_out_2.shape[0] == 6
    assert vae_decoder_out_3.shape[0] == 3

    assert torch.all(vae_decoder_out_1 == vae_decoder_out_all[0:3])
    assert torch.all(vae_decoder_out_2 == vae_decoder_out_all[3:9])
    assert torch.all(vae_decoder_out_3 == vae_decoder_out_all[9:])

@torch.no_grad()
def test_text_encoder_swapping(
    tiny_sd, 
    cat_prompt,
    panda_prompt,
    birthday_cake_prompt,
    wave_prompt
):

    module = tiny_sd.text_encoder.text_model.encoder.layers[-1]

    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):

            module.output = (torch.ones_like(module.output[0]) * 0,)
            encoder_out_1 = module.output[0].save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            module.output = (torch.ones_like(module.output[0]) * 1,)
            encoder_out_2 = module.output[0].save()

        with tracer.invoke(wave_prompt):
            module.output = (torch.ones_like(module.output[0]) * 2,)
            encoder_out_3 = module.output[0].save()

        with tracer.invoke():
            encoder_out_all = module.output[0].save()

    assert encoder_out_all.shape[0] == encoder_out_1.shape[0] + encoder_out_2.shape[0] + encoder_out_3.shape[0]
    assert encoder_out_1.shape[0] == 1
    assert encoder_out_2.shape[0] == 2
    assert encoder_out_3.shape[0] == 1

    assert torch.all(encoder_out_1 == encoder_out_all[0:1])
    assert torch.all(encoder_out_2 == encoder_out_all[1:3])
    assert torch.all(encoder_out_3 == encoder_out_all[3:])
    

@torch.no_grad()
def test_unet_swapping(
    tiny_sd, 
    cat_prompt,
    panda_prompt,
    birthday_cake_prompt,
    wave_prompt
):

    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        guidance_scale=1.0,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):
            tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 0,)
            unet_out_1 = tiny_sd.unet.output[0].save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 1,)
            unet_out_2 = tiny_sd.unet.output[0].save()

        with tracer.invoke(wave_prompt):
            tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 2,)
            unet_out_3 = tiny_sd.unet.output[0].save()

        with tracer.invoke():
            unet_out_all = tiny_sd.unet.output[0].save()

    assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
    assert unet_out_1.shape[0] == 3
    assert unet_out_2.shape[0] == 6
    assert unet_out_3.shape[0] == 3

    assert torch.all(unet_out_1 == unet_out_all[0:3])
    assert torch.all(unet_out_2 == unet_out_all[3:9])
    assert torch.all(unet_out_3 == unet_out_all[9:])


@torch.no_grad()
def test_unet_swapping_with_guidance(
    tiny_sd, 
    cat_prompt,
    panda_prompt,
    birthday_cake_prompt,
    wave_prompt
):

    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        guidance_scale=7.5,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):
            tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 0,)
            unet_out_1 = tiny_sd.unet.output[0].save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 1,)
            unet_out_2 = tiny_sd.unet.output[0].save()

        with tracer.invoke(wave_prompt):
            tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 2,)
            unet_out_3 = tiny_sd.unet.output[0].save()

        with tracer.invoke():
            unet_out_all = tiny_sd.unet.output[0].save()

    assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
    assert unet_out_1.shape[0] == 6
    assert unet_out_2.shape[0] == 12
    assert unet_out_3.shape[0] == 6

    assert torch.all(unet_out_1 == torch.cat([unet_out_all[0:3], unet_out_all[12:15]], dim=0))
    assert torch.all(unet_out_2 == torch.cat([unet_out_all[3:9], unet_out_all[15:21]], dim=0))
    assert torch.all(unet_out_3 == torch.cat([unet_out_all[9:12], unet_out_all[21:24]], dim=0))


@torch.no_grad()
def test_vae_swapping(
    tiny_sd,
    cat_prompt,
    panda_prompt,
    birthday_cake_prompt,
    wave_prompt
):
    
    with tiny_sd.generate(
        num_inference_steps=20,
        num_images_per_prompt=3,
        guidance_scale=1.0,
        seed=423
    ) as tracer:

        with tracer.invoke(cat_prompt):
            tiny_sd.vae.decoder.output = torch.ones_like(tiny_sd.vae.decoder.output) * 0
            unet_out_1 = tiny_sd.vae.decoder.output.save()

        with tracer.invoke([panda_prompt, birthday_cake_prompt]):
            tiny_sd.vae.decoder.output = torch.ones_like(tiny_sd.vae.decoder.output) * 1
            unet_out_2 = tiny_sd.vae.decoder.output.save()

        with tracer.invoke(wave_prompt):
            tiny_sd.vae.decoder.output = torch.ones_like(tiny_sd.vae.decoder.output) * 2
            unet_out_3 = tiny_sd.vae.decoder.output.save()

        with tracer.invoke():
            unet_out_all = tiny_sd.vae.decoder.output.save()

    assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
    assert unet_out_1.shape[0] == 3
    assert unet_out_2.shape[0] == 6
    assert unet_out_3.shape[0] == 3

    assert torch.all(unet_out_1 == unet_out_all[0:3])
    assert torch.all(unet_out_2 == unet_out_all[3:9])
    assert torch.all(unet_out_3 == unet_out_all[9:])
