import os
import subprocess
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import click
import dnnlib
import numpy as np
import pickle
import tqdm
import torch
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    ode_list = []
   
    x_next = latents.to(torch.float64) * t_steps[0]
    ode_list.append(x_next)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

       
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

       
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        ode_list.append(x_next)
    return x_next,ode_list

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_gpu_ids(gpu_id_value):
    gpu_ids = []
    for part in gpu_id_value.split(','):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise click.ClickException('--gpu_id must be an integer or a comma-separated list like 0,1,2,3.')
        gpu_ids.append(int(part))

    if not gpu_ids:
        raise click.ClickException('--gpu_id cannot be empty.')
    if len(set(gpu_ids)) != len(gpu_ids):
        raise click.ClickException('--gpu_id contains duplicate GPU ids.')
    return gpu_ids


def build_worker_command(network_pkl, save_dir, class_idx, gpu_id, worker_index, worker_count, num_steps, sigma_min, sigma_max, rho, S_churn, S_min, S_max, S_noise):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        '--network',
        network_pkl,
        '--save_dir',
        save_dir,
        '--gpu_id',
        str(gpu_id),
        '--worker_index',
        str(worker_index),
        '--worker_count',
        str(worker_count),
        '--steps',
        str(num_steps),
        '--rho',
        str(rho),
        '--S_churn',
        str(S_churn),
        '--S_min',
        str(S_min),
        '--S_max',
        str(S_max),
        '--S_noise',
        str(S_noise),
    ]

    if class_idx is not None:
        command.extend(['--class', str(class_idx)])
    if sigma_min is not None:
        command.extend(['--sigma_min', str(sigma_min)])
    if sigma_max is not None:
        command.extend(['--sigma_max', str(sigma_max)])
    return command

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--save_dir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--gpu_id',                  help='Current GPU id, or a comma-separated list like 0,1,2,3', metavar='INT|LIST', type=str, default='0', show_default=True)
@click.option('--num_gpus',                help='Total number of GPUs used for sharding', metavar='INT',            type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--worker_index',            type=click.IntRange(min=0), default=None, hidden=True)
@click.option('--worker_count',            type=click.IntRange(min=1), default=None, hidden=True)
#----------------------------------------------------------------------------

def main(network_pkl, save_dir, class_idx, gpu_id, num_gpus, num_steps, sigma_min, sigma_max, rho, S_churn, S_min, S_max, S_noise, worker_index, worker_count):
    gpu_ids = parse_gpu_ids(gpu_id)

    if len(gpu_ids) > 1 and worker_index is None:
        processes = []
        total_workers = len(gpu_ids)
        print(f'Launching {total_workers} workers on GPUs {gpu_ids}')
        for index, actual_gpu_id in enumerate(gpu_ids):
            command = build_worker_command(
                network_pkl=network_pkl,
                save_dir=save_dir,
                class_idx=class_idx,
                gpu_id=actual_gpu_id,
                worker_index=index,
                worker_count=total_workers,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                S_churn=S_churn,
                S_min=S_min,
                S_max=S_max,
                S_noise=S_noise,
            )
            print(f'Starting worker {index} on GPU {actual_gpu_id}')
            processes.append((actual_gpu_id, subprocess.Popen(command)))

        failed_gpu_ids = []
        for actual_gpu_id, process in processes:
            if process.wait() != 0:
                failed_gpu_ids.append(actual_gpu_id)
        if failed_gpu_ids:
            raise click.ClickException(f'Workers failed on GPUs: {failed_gpu_ids}')
        return

    actual_gpu_id = gpu_ids[0]
    shard_index = worker_index if worker_index is not None else actual_gpu_id
    shard_count = worker_count if worker_count is not None else num_gpus

    if shard_index >= shard_count:
        raise click.ClickException('Shard index must be smaller than shard count.')
    if len(gpu_ids) == 1 and worker_index is None and actual_gpu_id >= num_gpus:
        raise click.ClickException('--gpu_id must be smaller than --num_gpus when using a single GPU id.')

    device = torch.device(f'cuda:{actual_gpu_id}')
    
    os.makedirs(save_dir, exist_ok=True)

    sampler_kwargs = {
        'num_steps': num_steps,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'rho': rho,
        'S_churn': S_churn,
        'S_min': S_min,
        'S_max': S_max,
        'S_noise': S_noise,
    }
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}

    all_seeds = list(range(200000))
    sampled_seeds = all_seeds[shard_index::shard_count]
    output_path = os.path.join(save_dir, f'trajectory_gpu{actual_gpu_id}.pt')

    print(f'GPU {actual_gpu_id}: processing {len(sampled_seeds)} seeds out of {len(all_seeds)} total.')
    print(f'Saving shard to "{output_path}"')

    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    with open(output_path, 'wb') as output_file:
        pickle.dump(
            {
                'gpu_id': actual_gpu_id,
                'shard_index': shard_index,
                'num_gpus': shard_count,
                'num_seeds': len(sampled_seeds),
                'seeds': sampled_seeds,
                'sampler_kwargs': sampler_kwargs,
            },
            output_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for i, seed in enumerate(tqdm.tqdm(sampled_seeds, desc="Generating")):
            rnd = StackedRandomGenerator(device, [seed])
            latents = rnd.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = None
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[1], device=device)]
            if class_idx is not None and class_labels is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1
            elif class_idx is not None:
                raise click.ClickException('The selected network is unconditional and does not support --class.')

            _, ode_list = edm_sampler(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
            image_all = torch.cat(ode_list, dim=0).cpu()
            pickle.dump(
                {
                    'seed': seed,
                    'trajectory': image_all,
                },
                output_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

            if i % 100 == 0:
                print(f"Saved {i} image pairs")

if __name__ == "__main__":
    main()