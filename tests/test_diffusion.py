"""
Tests for DiffusionModel functionality.

These tests cover diffusion model specific features:
- Image generation
- Tracing and interventions on diffusion models
"""

import pytest
import torch
import PIL

if not torch.cuda.is_available():
    pytest.skip("no GPU available.", allow_module_level=True)

from nnsight.modeling.diffusion import DiffusionModel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tiny_sd():
    """Load tiny stable diffusion model."""
    return DiffusionModel(
        "segmind/tiny-sd", torch_dtype=torch.float16, dispatch=True
    ).to("cuda")


@pytest.fixture(scope="module")
def cat_prompt():
    """Cat image generation prompt."""
    return "A brown and white cat staring off with pretty green eyes"


# =============================================================================
# Generation Tests
# =============================================================================


class TestGeneration:
    """Tests for diffusion model image generation."""

    def test_basic_generation(self, tiny_sd, cat_prompt):
        """Test basic image generation without tracing."""
        num_images_per_prompt = 3

        images = tiny_sd.generate(
            cat_prompt,
            num_inference_steps=20,
            num_images_per_prompt=num_images_per_prompt,
            seed=423,
            trace=False,
        ).images

        assert len(images) == num_images_per_prompt
        assert all([type(img) == PIL.Image.Image for img in images])
        assert all(
            images[i] != images[j]
            for i in range(num_images_per_prompt)
            for j in range(i + 1, num_images_per_prompt)
        )
