import argparse
import os
import re
import socket
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import lpips
import random
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from fid_computation import eval_fid
from model import ModifiedEDMPrecond
import dnnlib
import pickle


save_dir = "HKDiffusion/data_preparation"

class CIFARPairBatchDataset(Dataset):
    def __init__(self, batch_dir, gpu_ids):
        self.batch_files = [
            os.path.join(batch_dir, f"trajectory_gpu{gpu_id}.pt")
            for gpu_id in gpu_ids
        ]
        self.data_list = []

        for batch_file in self.batch_files:
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"Trajectory file not found: {batch_file}")

            with open(batch_file, 'rb') as f:
                header = pickle.load(f)
                if not isinstance(header, dict) or 'gpu_id' not in header:
                    raise ValueError(f"Invalid trajectory header in {batch_file}")

                while True:
                    try:
                        sample = pickle.load(f)
                    except EOFError:
                        break

                    if not isinstance(sample, dict) or 'trajectory' not in sample:
                        raise ValueError(f"Invalid trajectory record in {batch_file}")
                    self.data_list.append(sample['trajectory'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_available_gpu_ids(batch_dir):
    gpu_ids = []
    for file_name in os.listdir(batch_dir):
        match = re.fullmatch(r'trajectory_gpu(\d+)\.pt', file_name)
        if match:
            gpu_ids.append(int(match.group(1)))
    return sorted(gpu_ids)


def t_sample(
    i, device, u_net,
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

    t_cur = t_steps[i]
        
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    
    return t_hat


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]

def train(rank,args):

    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.master_port}',
        world_size=args.world_size,  
        rank=rank  
    )
    
    # Set GPU for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank )

    checkpoint_dir = "HKDiffusion/checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    batch_size = args.batch_size
    
    pkl_path = "HKDiffusion/edm-cifar10-32x32-uncond-vp.pkl"
    u_net = ModifiedEDMPrecond.from_pkl(pkl_path)

    
    with dnnlib.util.open_url(pkl_path, verbose=True) as f:
        edm_net = pickle.load(f)['ema'].to(device)
    
    
    u_net = u_net.to(device)
    u_net = DDP(u_net, device_ids=[rank],find_unused_parameters=True)
    

    a_shapes = (
    [(1, 128, 32 * 32)] +
    [(1, 256, 32 * 32)] * 4 +
    [(1, 256, 16 * 16)] * 5 +
    [(1, 256, 8 * 8)] * 5
    )

    a_list = [
        torch.zeros(shape, device=device).requires_grad_(True)
        for shape in a_shapes
    ]

    

    
    optimizer_u_net = optim.Adam(u_net.parameters(), lr=args.lr_u_net)
    optimizer_a = optim.Adam(a_list, lr=args.lr_a)
    scheduler_u_net = ExponentialLR(optimizer_u_net, gamma=args.gamma)
    scheduler_a = ExponentialLR(optimizer_a, gamma=args.gamma)


    epoch, init_epoch = 0, 0
    total_epochs = max(args.num_epoch, 1)
    data_loader = None
    dataset = None
    sampler = None
    
    
    gpu_id_groups = [[gpu_id] for gpu_id in get_available_gpu_ids(save_dir)]
    if not gpu_id_groups:
        raise FileNotFoundError(f"No trajectory_gpu*.pt files found in {save_dir}")

    for epoch in range(init_epoch, args.num_epoch + 1):
        for sub_epoch, gpu_ids in enumerate(gpu_id_groups):
            if data_loader is not None: del data_loader
            if sampler is not None: del sampler
            if dataset is not None: del dataset
            torch.cuda.empty_cache()
            dataset = CIFARPairBatchDataset(save_dir, gpu_ids=gpu_ids)
            sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=False)
    
            sampler.set_epoch(epoch * len(gpu_id_groups) + sub_epoch)
            u_net.train()
        
            for iteration, x in enumerate(data_loader):
                
                noisy_image = x.to(device)
                optimizer_u_net.zero_grad()
                optimizer_a.zero_grad()
            
                
                mse = nn.MSELoss()
                loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
            
                x0 = noisy_image[:,-1,:,:,:]
                
                torch.distributed.barrier()
                t_list = [0]
                numbers = random.sample(range(0, args.num_inference_steps), 4)
                t_list.extend(numbers)


                loss_list_all = torch.tensor(0.).to(device)
                for i in t_list:
                
                    xt = noisy_image[:,i,:,:,:]
                    if not isinstance(xt, torch.Tensor):
                        xt = torch.tensor(xt)
                    xt = xt.to(device)
                
                    t = torch.tensor(args.num_inference_steps-i)

                    sigma = t_sample(i,device,edm_net)
                    sigma_final = t_sample(-2,device,edm_net)
                    x0_rec, up_sample_list_rec = u_net(xt, a_list, t, sigma, sigma_final)

                    loss = 1e-3 ** (epoch / total_epochs) * mse(x0_rec.float(),x0.float()) + loss_fn_vgg(x0_rec.float(), x0.float()).mean() 

                    torch.distributed.barrier()
                    
                    torch.cuda.empty_cache()

                    loss_list_all += loss 
                
                
                torch.distributed.barrier()
                loss =  loss_list_all/len(t_list)
                loss = loss.float()
                
                print(f"Epoch {epoch+1} | Sub_epoch {sub_epoch+1} | Batch_idx {iteration+1}/{len(data_loader)} | Train Loss: {loss.item()} ")
                
                if (epoch < 40 and epoch % 5 == 0 and iteration % 50 == 0) or (epoch >= 40 and iteration % 50 == 0):
                    with torch.no_grad():
                        eval_fid(u_net.module,a_list,sigma,sigma_final,device)
                        torch.distributed.barrier()

                loss.backward()
                
                optimizer_u_net.step()
                optimizer_a.step()
                torch.distributed.barrier()

                if rank == 0:
                    torch.save({
                    'u_net': u_net.state_dict(),
                    'a_list': [a.detach().cpu() for a in a_list]
                    }, checkpoint_path)
                torch.distributed.barrier()

            torch.distributed.barrier()
            
        scheduler_u_net.step()
        scheduler_a.step()
        torch.distributed.barrier()
        min_lr = 1e-6
        for param_group in optimizer_u_net.param_groups:
            if param_group['lr'] < min_lr:
            
                param_group['lr'] = min_lr
                
        
        for param_group in optimizer_a.param_groups:
            if param_group['lr'] < min_lr:
                
                param_group['lr'] = min_lr
                    
       
           



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)

    parser.add_argument('--lr_u_net', type=float,
                        default=1e-3, help='learning rate g')
    parser.add_argument('--lr_a', type=float, default=1e-3,
                        help='learning rate d')
    parser.add_argument('--gamma', type=float, default=0.99, help='Learning rate decay rate for ExponentialLR')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--num_inference_steps', type=int, default=18, help='Number of inference steps')
    
    args = parser.parse_args()
    
    args.world_size = args.num_gpus  
    args.master_port = find_free_port()

    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
