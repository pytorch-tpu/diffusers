from diffusers import DiffusionPipeline
from time import time
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met


server = xp.start_server(9012)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9",
                                        #  torch_dtype=torch.bfloat16,
                                         use_safetensors=True,
                                        #  variant="bf16"
                                        )
device = xm.xla_device()
pipe.to(device)

bs = 8
inference_steps = 40
height = width = 512

prompts = ["a photo of an astronaut riding a horse on mars"] * bs
print(f'batch size = {bs}, inference steps = {inference_steps}',
      f'height = width = {width}',
      flush=True
      )

iters = 15
print('starting inference', flush=True)
for i in range(iters):
    start = time()
    prompt = prompts
    image = pipe(prompt,
                 num_inference_steps=inference_steps,
                 height=height,
                 width=width,
                 ).images[0]
    print(f'Step {i} inference time {time()-start} sec', flush=True)

