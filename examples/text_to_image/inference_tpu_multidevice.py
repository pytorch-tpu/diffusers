import os
import sys
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
from time import time
from typing import Tuple
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def setup_model_parallel() -> Tuple[int, int]:
    # assuming model parallelism over the whole world size
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    # seed must be the same in all processes
    torch.manual_seed(1)
    device = xm.xla_device()
    xm.set_rng_state(1, device=device)
    return rank, world_size


def main(index):
    # starts server at port 9012 for profiling
    # (refer to https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    server = xp.start_server(9012)
    device = xm.xla_device()
    rank, world_size = setup_model_parallel()

    # print only for xla:0 device
    if rank > 0:
        sys.stdout = open(os.devnull, "w")


    model_id = "stabilityai/stable-diffusion-2-1" # xp.spawn fails for sd-xl model
    # Uses the default DPMSolverMultistepScheduler (DPM-Solver++) scheduler
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                  torch_dtype=torch.bfloat16,
                                                  )
    pipe = pipe.to(device)

    global_bs = 64
    inference_steps = 50
    prompts = ["a photo of an astronaut riding a horse on mars"] * global_bs
    print(f'global batch size {global_bs}',
          f'inference steps {inference_steps}',
          flush=True
          )

    iters = 5
    for i in range(iters):
        start = time()
        # prompt = prompts[rank::world_size]
        prompt = prompts[rank]
        image = pipe(prompt, num_inference_steps=inference_steps).images[0]
        print(f'Step {i} inference time {time()-start} sec', flush=True)


if __name__ == '__main__':
    xmp.spawn(main, args=())
