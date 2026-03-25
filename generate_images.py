from diffusers import StableDiffusionPipeline
import torch
import os

def generate_images():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cpu")

    prompts = [
        "delivery truck in city, marketing photo",
        "package delivery service, modern logistics",
        "happy customer receiving package"
    ]

    os.makedirs("output", exist_ok=True)

    images = []

    for i, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        path = f"output/image{i}.png"
        image.save(path)
        images.append(path)

    return images
