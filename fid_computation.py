"""Compute FID by comparing the statistics of generated images with those of reference data."""

import os

os.environ['HF_HUB_URL'] = 'https://hf-mirror.com'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import socket
import re
import json
import click
import scipy.linalg
import torch
import dnnlib

from torch_utils import distributed as dist
from training import dataset

import click
import tqdm
import pickle
import numpy as np

import PIL.Image


#----------------------------------------------------------------------------
def t_sample(
     device, u_net, 
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1
):
    device = device
    net = u_net
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    t_cur = t_steps[0]
        
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    
    return t_cur,t_hat

def t_sample_final(
     device, u_net, 
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1
):
    device = device
    net = u_net
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    t_cur = t_steps[-2]
        
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    
    return t_hat

def sampler(
    u_net, a_list, latents, class_labels = None, randn_like=torch.randn_like
):

    
    pkl_path = "HKDiffusion/edm-cifar10-32x32-uncond-vp.pkl"
    with dnnlib.util.open_url(pkl_path, verbose=True) as f:
        edm_net = pickle.load(f)['ema'].to(latents.device)
    
    t_cur,t_hat = t_sample(latents.device,edm_net)
    
    sigma = t_hat
    sigma_final = t_sample_final(latents.device,edm_net)
    z = latents.to(torch.float32) * t_cur
    
    
    i = 0
    t = torch.tensor(18-i)

    x0_rec, _ = u_net(z, a_list, t, sigma, sigma_final,class_labels)
    return x0_rec

def calculate_inception_stats(detector_url,
    image_path, device,num_expected=None, seed=0, max_batch_size=32,
    num_workers=0, prefetch_factor=None
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    #detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

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

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

    
class CommaSeparatedList(click.ParamType):
    
    name = 'list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')


def save_image(img, num_channel, fname):
    assert num_channel in [1, 3]
    if num_channel == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if num_channel == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def save_fid(fid, fname):        
    formatted_string = f'{fid:g}'  # Format the number using :g specifier
    # Open the file in write mode and save the formatted string
    # with open(fname, 'w') as file:
    #     file.write(formatted_string)  
    with open(fname, 'a') as file:
        file.write(formatted_string) 
        
#----------------------------------------------------------------------------

def eval_fid(u_net, a_list, sigma,sigma_final,device,init_sigma=1., data_stat=None, outdir='image_experiment/out',
             detector_url='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl',
             seeds='0-49999', subdirs=False, max_batch_size=32, ref_path='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'):
    
    opts = dnnlib.EasyDict(
        init_sigma=init_sigma,
        data_stat=data_stat,
        outdir=outdir,
        detector_url=detector_url,
        seeds=seeds,
        subdirs=subdirs,
        max_batch_size=max_batch_size,
        ref_path=ref_path
    )
   
    seeds = parse_int_list(seeds)
    max_batch_size = max_batch_size
    
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()


    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        
        img_channels = 3
        img_resolution = 32
        latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)
        # class_labels = None
        # if net.label_dim:
        #     class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        # if class_idx is not None:
        #     class_labels[:, :] = 0
        #     class_labels[:, oclass_idx] = 1

        # Generate images.
        sampler_kwargs = {}
        images = sampler(u_net.to(device), [tensor.to(device) for tensor in a_list],  latents,  randn_like=rnd.randn_like, **sampler_kwargs)
        # print("images",images.shape,images.max(),images.min())
        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        image_dir = outdir
        os.makedirs(image_dir, exist_ok=True)
        for seed, image_np in zip(batch_seeds, images_np):
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            save_image(img=image_np,num_channel=image_np.shape[2],fname=image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done Generating images.')

    dist.print0('Computing FID')
   
    image_path = outdir
    ref_path = ref_path
    detector_url=detector_url
    num_expected = len(seeds)
    seed = 0

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(detector_url=detector_url,image_path=image_path, device = device,num_expected=num_expected, seed=seed, max_batch_size=max_batch_size)
    dist.print0('Calculating FID...')
    fid = torch.tensor(0.0, dtype=torch.float32, device='cuda') 

    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'{fid:g}')
    
    
       
        txt_path = os.path.join(os.path.dirname(opts.outdir),f'fid.txt')
        save_fid(fid=fid,fname=txt_path)
        
    
    torch.distributed.barrier()
    dist.print0('Done Computing FID')
    