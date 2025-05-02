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
