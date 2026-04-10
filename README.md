# HKDiffusion

Official implementation of the paper Fast Generation with Interpretable Diffusion Trajectory.

This repository studies a trajectory-based training pipeline built on top of a pretrained EDM model. Instead of training directly from standard image folders, the project first generates full denoising trajectories with a frozen EDM sampler, stores them as GPU-sharded `.pt` files, and then trains the reconstruction model on these trajectories.

The two most important parts of the project are:

1. Data preparation: generate trajectory files from a pretrained EDM checkpoint.
2. Training: load the generated trajectory shards and optimize the HKDiffusion model.

## Project Structure

- [train.py](HKDiffusion/train.py): main multi-GPU training entry.
- [data_preparation/trajectory_generation.py](HKDiffusion/data_preparation/trajectory_generation.py): trajectory generation script.
- [fid_computation.py](HKDiffusion/fid_computation.py): FID evaluation utilities used during training.
- [model.py](HKDiffusion/model.py): main model definition.
- [dnnlib](HKDiffusion/dnnlib): EDM utility package.
- [torch_utils](HKDiffusion/torch_utils): EDM training and distributed utility package.
- [data_preparation](HKDiffusion/data_preparation): generated trajectory shards are stored here.
- [checkpoints](HKDiffusion/checkpoints): training checkpoints are written here.

## Running the Code

The project contains two main stages: trajectory generation and model training.

### 0. Download the Pretrained EDM Checkpoint

You need to download [edm-cifar10-32x32-uncond-vp.pkl](HKDiffusion/edm-cifar10-32x32-uncond-vp.pkl) manually into the HKDiffusion folder before running the code.

Run:

```bash
wget -O HKDiffusion/edm-cifar10-32x32-uncond-vp.pkl \
  https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

After downloading, the checkpoint should be located at [edm-cifar10-32x32-uncond-vp.pkl](HKDiffusion/edm-cifar10-32x32-uncond-vp.pkl).

### 1. Prepare Trajectory Data

The script [data_preparation/trajectory_generation.py](HKDiffusion/data_preparation/trajectory_generation.py) uses the pretrained EDM checkpoint [edm-cifar10-32x32-uncond-vp.pkl](HKDiffusion/edm-cifar10-32x32-uncond-vp.pkl) to generate full denoising trajectories and save them into GPU-sharded `.pt` files under [data_preparation](HKDiffusion/data_preparation).

Run:

```bash
python HKDiffusion/data_preparation/trajectory_generation.py \
  --network HKDiffusion/edm-cifar10-32x32-uncond-vp.pkl \
  --save_dir HKDiffusion/data_preparation \
  --gpu_id 0,1,2,3
```

After this step, files such as `trajectory_gpu0.pt`, `trajectory_gpu1.pt`, `trajectory_gpu2.pt`, and `trajectory_gpu3.pt` will be created in [data_preparation](HKDiffusion/data_preparation).

### 2. Train HKDiffusion

The training script [train.py](HKDiffusion/train.py) reads the generated `trajectory_gpu*.pt` files, launches distributed training with DDP, evaluates FID during training, and saves checkpoints to [checkpoints](HKDiffusion/checkpoints).

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python HKDiffusion/train.py --num_gpus 4 --batch_size 48
```

If you want to use another set of physical GPUs, first set `CUDA_VISIBLE_DEVICES`, and the program will use local ids `0,1,2,3` internally.

## Training Time

On CIFAR-10, our method takes about two days to train on 8 NVIDIA V100 GPUs.
