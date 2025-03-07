import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate(dest):
    directory = os.path.dirname(dest)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to(device)
    width, height = 304, 200  
    prompt = "Andy Warhol-style pop art portrait with bold patterns, bright neon colors like pink, yellow, and blue. The subject is a famous icon like a flower or packaging, with repetitive color blocking, thick black outlines, and a minimal background.(NO NSFW)"
    with torch.no_grad():
        image = pipe(prompt, height=height, width=width, guidance_scale=7.5).images[0]
    image = image.crop((0, 0, 300, 200))
    image.save(dest)

