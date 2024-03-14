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
    default=2, # 8,
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
    default=2, # 30,
    help='Number of itterations to run the benchmark.'
    )

    return parser.parse_args(args)


def main(args):
    server = xp.start_server(9012)
    # pipe = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-0.9",
    #     use_safetensors=True,
    #     )
    device = xm.xla_device()
    # pipe.to(device)

    bs = args.batch_size # 1
    inference_steps = args.inf_steps # 2
    height = width = args.width # 512

    prompts = ["a photo of an astronaut riding a horse on mars"] * bs
    print(f'batch size = {bs}, inference steps = {inference_steps}',
          f'height = width = {width}',
          flush=True
          )

    import torch
    import torch_xla.experimental.fori_loop
    from torch._higher_order_ops.while_loop import while_loop
    def cond_fn(init, limit_value):
      return limit_value[0] <= init[0]

    def body_fn(init, limit_value):
      one_value = torch.ones(1, dtype=torch.int32, device=device)
      two_value = limit_value.clone()
      # start = time()
      pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-0.9",
        use_safetensors=True,
        )
    # device = xm.xla_device()
      pipe.to(device)
      image = pipe(["a photo of an astronaut riding a horse on mars"], # prompts,
                  num_inference_steps=2, # inference_steps,
                  height=512, # height,
                  width=512, # width,
                  ).images[0]
      print("type of image: ", type(image))
    #   print(f'Step {i} inference time {time()-start} sec', flush=True)
      return (torch.sub(init, one_value), two_value)
    
    start = time()
    # iters = 3
    init = torch.tensor([3], dtype=torch.int32, device=device)
    limit_value = torch.tensor([0], dtype=torch.int32, device=device)
    # res = while_loop(cond_fn, body_fn, (init, limit_value))
    from torch_xla.experimental.fori_loop import _xla_while_loop
    res = _xla_while_loop(cond_fn, body_fn, (init, limit_value))
    print(f'Call pipeline with _xla_while_loop for three times used {time()-start} sec', flush=True)
    print("result of while_loop: ", res)
    # expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    # self.assertEqual(expected, res)

    # start2 = time()
    # iters = 3
    # for i in range(iters):
    #     pipe2 = DiffusionPipeline.from_pretrained(
    #         "stabilityai/stable-diffusion-xl-base-0.9",
    #         use_safetensors=True,
    #         )
    # print(f'Call pipeline without _xla_while_loop for three times used {time()-start2} sec', flush=True)

    # iters = 1 # 15
    # print('starting inference', flush=True)
    # for i in range(iters):
    #     start = time()
    #     image = pipe(prompts,
    #                 num_inference_steps=inference_steps,
    #                 height=height,
    #                 width=width,
    #                 ).images[0]
    #     print(f'Step {i} inference time {time()-start} sec', flush=True)


if __name__ == '__main__':
  args = parser(sys.argv[1:])
  main(args)
