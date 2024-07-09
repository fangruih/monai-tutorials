# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging

import os
import sys
from pathlib import Path

import torch
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import KL_loss, define_instance, prepare_dataloader, prepare_dataloader_extract_dataset, prepare_dataloader_extract_dataset_custom, setup_ddp, print_gpu_memory
from test_utils import *
from visualize_image import visualize_one_slice_in_3d_image
from create_dataset import *
import wandb


def main():
    parser = argparse.ArgumentParser(description="PyTorch VAE-GAN training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()
    wandb.init(project=args.wandb_project_name, config=args)
    wandb.config.update(args)


    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    
    base_dir = '/home/sijun/meow/data_new/hcp/registered'
    # hcp_dataset, _ = create_train_val_datasets(base_dir)
    
    
    
    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["num_channels"]) - 1)
    hcp_dataset,val_dataset =  prepare_dataloader_extract_dataset_custom(
        args,
        args.autoencoder_train["batch_size"],
        args.autoencoder_train["patch_size"],
        base_dir= base_dir,
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=False,
        size_divisible=size_divisible,
        amp=False,
    )
    print("done hcp")
    decathlon_dataset_train,decathlon_dataset_val = prepare_dataloader_extract_dataset(
        args,
        args.autoencoder_train["batch_size"],
        args.autoencoder_train["patch_size"],
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=False,
        size_divisible=size_divisible,
        amp=False,
    )
    
    
    
    print("done decathlon ")
    
    
    
    compare_datasets(decathlon_dataset_train,decathlon_dataset_val)
    print("done compare")

if __name__ == "__main__":
    
    main()
