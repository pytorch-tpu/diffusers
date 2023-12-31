import argparse
import sys
from time import time
from diffusers import DiffusionPipeline
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met


def parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--batch-size',
    type=int,
    default=8,
    help='Number of images to generate'
    )

    parser.add_argument(
    '--width',
    type=int,
    default=512,
    help='Width'
    )

    parser.add_argument(
    '--inf-steps',
    type=int,
    default=30,
    help='Number of itterations to run the benchmark.'
    )

    return parser.parse_args(args)


def main(args):
    server = xp.start_server(9012)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-0.9",
        use_safetensors=True,
        )
    device = xm.xla_device()
    pipe.to(device)

    bs = args.batch_size
    inference_steps = args.inf_steps
    height = width = args.width

    prompts = ["a photo of an astronaut riding a horse on mars"] * bs
    print(f'batch size = {bs}, inference steps = {inference_steps}',
          f'height = width = {width}',
          flush=True
          )

    iters = 15
    print('starting inference', flush=True)
    for i in range(iters):
        start = time()
        image = pipe(prompts,
                    num_inference_steps=inference_steps,
                    height=height,
                    width=width,
                    ).images[0]
        print(f'Step {i} inference time {time()-start} sec', flush=True)


if __name__ == '__main__':
  args = parser(sys.argv[1:])
  main(args)
