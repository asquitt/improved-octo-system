#!/usr/bin/env python3
"""
Week 2 Starter Code: Distributed Data Parallel (DDP)

TODO: Convert single-GPU code to multi-GPU with DDP
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """
    Initialize distributed training

    TODO: Set up process group
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # TODO: Initialize process group
    dist._______(backend='______', rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """
    Main DDP training function

    TODO: Implement DDP training loop
    """
    # TODO: Setup
    setup(rank, world_size)

    # TODO: Set device for this rank
    device = torch.device(f'cuda:{______}')

    # TODO: Create model and wrap in DDP
    model = MyModel().to(device)
    model = ______(model, device_ids=[______])

    # TODO: Create DistributedSampler for data loading
    sampler = ______(train_dataset, num_replicas=world_size, rank=rank)

    # TODO: Training loop (similar to single-GPU)
    # ...

    cleanup()

def main():
    world_size = torch.cuda.device_count()

    # TODO: Spawn processes for each GPU
    mp.______(
        train_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
