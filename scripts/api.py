## Installation
# Save this file to the InvokeAI/scripts directory as api.py
# In the root directory run:
# pip3 install wheel fastapi uvicorn
# uvicorn scripts.api:app --host 0.0.0.0 --port 9090 --reload

## Test
# curl -X 'POST' http://localhost:9090/generate -H 'Content-Type: application/json' -d '{ "prompt": "New Student" }' -o image.png

import os
import io
import random
from typing import Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel
import transformers
transformers.logging.set_verbosity_error()
from ldm.generate import Generate
from ldm.invoke.restoration import Restoration

app = FastAPI()
restoration = Restoration()
gfpgan, codeformer = restoration.load_face_restore_models('./src/gfpgan/experiments/pretrained_models/GFPGANv1.4.pth')
esrgan = restoration.load_esrgan(400)
generator = Generate(
    gfpgan = gfpgan,
    codeformer = codeformer,
    esrgan = esrgan
)
if not os.path.exists('outputs/img-samples'):
    os.makedirs('outputs/img-samples')
generator.load_model()

class Params(BaseModel):
    prompt: str
    seed: Optional[int] = None
    steps: Optional[int] = 50
    cfg_scale: Optional[float] = 7.5
    threshold: Optional[float] = 0
    perlin: Optional[float] = 0
    height: Optional[int] = 512
    width: Optional[int] = 512
    sampler: Optional[str] = 'k_lms'
    esrgan_postprocess: Optional[bool] = False
    esrgan_scale: Optional[int] = 4
    esrgan_strength: Optional[float] = 0.75
    esrgan_seed: Optional[int] = None
    gfpgan_postprocess: Optional[bool] = False
    gfpgan_strength: Optional[float] = 0.8
    gfpgan_seed: Optional[int] = None

@app.post('/generate')
def post(params: Params):
    global generator, gfpgan, codeformer, esrgan
    seed = params.seed or random.randint(1, pow(2, 31) - 1 + pow(2, 31))
    generated_image = None
    def image_callback(image, *args, **kwargs):
        nonlocal generated_image
        generated_image = image
    generator.prompt2image(
        prompt = params.prompt,
        seed = seed,
        steps = params.steps,
        cfg_scale = params.cfg_scale,
        threshold = params.threshold,
        perlin = params.perlin,
        height = params.height,
        width = params.width,
        sampler = params.sampler,
        image_callback = image_callback
    )
    if params.esrgan_postprocess:
        generated_image = esrgan.process(
            image = generated_image,
            upsampler_scale = params.esrgan_scale,
            strength = params.esrgan_strength,
            seed = params.esrgan_seed or seed
        )
    if params.gfpgan_postprocess:
        generated_image = gfpgan.process(
            image = generated_image,
            strength = params.gfpgan_strength,
            seed = params.gfpgan_seed or seed
        )
    bytes = io.BytesIO()
    generated_image.save(bytes, format='PNG')
    content = bytes.getvalue()
    return Response(
        content = content,
        media_type = "image/png",
        headers={
            'Content-Disposition': 'attachment; filename="image.png"',
            'Content-Length': str(len(content))
        })