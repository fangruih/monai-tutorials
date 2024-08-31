import argparse
import json
import logging
import time

import torch.autograd.profiler as profiler

import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils import KL_loss, define_instance, prepare_dataloader, prepare_dataloader_extract_dataset_custom, setup_ddp, print_gpu_memory, prepare_file_list

from util.dataset_utils import prepare_dataloader_from_list
from plot_test.visualize_image import visualize_one_slice_in_3d_image, visualize_one_slice_in_3d_image_greyscale
from create_dataset import *
from metrics import metrics_mean_mses_psnrs_ssims_mmd
import wandb
from wandb import Image



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
#   if rank == 0:
#   print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
        
    from datetime import datetime
    # Generate a dynamic name based on the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'vae_{current_time}'
    if rank ==0:
        wandb.init(project=args.wandb_project_name_VAE,name=run_name, config=args)

    set_determinism(42)

    # Step 1: set data loader
    # Choose base directory base on the cluster storage. 
    # set up environment variable accordingly e.g. "export CLUSTER_NAME=sc" 
    cluster_name = os.getenv('CLUSTER_NAME')
    if cluster_name == 'vassar':
        base_dir = '/home/sijun/meow/data_new/'
    elif cluster_name == 'sc':
        # base_dir = '/simurgh/group/mri_data/'
        base_dir = '/scr/fangruih/mri_data/'
    else:
        raise ValueError('Unknown cluster name. Please set the CLUSTER_NAME environment variable. e.g. export CLUSTER=NAME=sc')

    size_divisible = 2 ** (len(args.autoencoder_def["num_channels"]) - 1)
    
    if args.dataset_type=="brain_tumor":
    
        train_loader, val_loader = prepare_dataloader(
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
        print("len(train_loader)", len(train_loader))
        print("len(val_loader)", len(val_loader))
    elif args.dataset_type=="hcp_ya_T1":
        
        train_loader, val_loader =  prepare_dataloader_extract_dataset_custom(
            args,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            # base_dir= base_dir,
            all_files= all_files, 
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            download=False,
            size_divisible=size_divisible,
            amp=False,
        )
    
    elif args.dataset_type=="T1_all":
        train_loader, val_loader =  prepare_dataloader_from_list(
            args,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            randcrop=args.autoencoder_train["random_crop"],
            rank=rank,
            world_size=world_size,
            cache=1.0,
            download=False,
            size_divisible=size_divisible,
            amp=False,
        )
        print(f'Number of batches in train_loader: {len(train_loader)}')
        print(f'Number of batches in val_loader: {len(val_loader)}')
    
    else: 
        raise ValueError(f"Unsupported dataset type specified: {args.dataset_type}")


    # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    def ensure_directory_exists(path):
        """ Ensure that the directory exists, and if not, create it. """
        os.makedirs(path, exist_ok=True)

    # Assuming `args.model_dir` and `current_time` are already defined
    args.model_dir = os.path.join(args.model_dir, current_time)
    
    # Ensure directories exist
    ensure_directory_exists(args.model_dir)
    trained_g_path_new = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path_new = os.path.join(args.model_dir, "discriminator.pt")
    trained_g_path_last_new = os.path.join(args.model_dir, "autoencoder_last.pt")
    trained_d_path_last_new = os.path.join(args.model_dir, "discriminator_last.pt")
    
    trained_g_path = os.path.join(args.autoencoder_dir,"autoencoder.pt")
    trained_d_path = os.path.join(args.autoencoder_dir, "discriminator.pt")
    trained_g_path_last = os.path.join(args.autoencoder_dir,"autoencoder_last.pt")
    trained_d_path_last = os.path.join(args.autoencoder_dir,"discriminator_last.pt")
    

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
            print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
        except:
            print(f"Rank {rank}: Train autoencoder from scratch.")

        try:
            discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location))
            print(f"Rank {rank}: Load trained discriminator from {trained_d_path}")
        except:
            print(f"Rank {rank}: Train discriminator from scratch.")

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    adv_weight = 0.01
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    # kl_weight: important hyper-parameter.
    #     If too large, decoder cannot recon good results from latent space.
    #     If too small, latent space will not be regularized enough for the diffusion model
    kl_weight = args.autoencoder_train["kl_weight"]

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["lr"] * world_size)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["lr"] * world_size)

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        # tensorboard_path = os.path.join(args.tfevent_path, "autoencoder")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # tensorboard_path = os.path.join(args.tfevent_path, "diffusion")
        tensorboard_path = os.path.join(args.tfevent_path, "autoencoder", current_time)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        
        print("autoencoder")
        print(autoencoder)
        # for param in autoencoder.module.encoder.parameters():
        #     param.requires_grad = False
        # # Detailed check of the encoder parameters
        # for name, param in autoencoder.module.named_parameters():
        #     print(f"Parameter: {name}, requires_grad={param.requires_grad}")


    # Step 4: training
    autoencoder_warm_up_n_epochs = 5
    n_epochs = args.autoencoder_train["n_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    intermediary_images = []
    n_example_images = 4
    best_val_recon_epoch_loss = 100.0
    total_step = 0
    

    for epoch in range(n_epochs):
        # train
        autoencoder.train()
        discriminator.train()
        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        # train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{n_epochs}")

        
        # # for step, batch in train_progress_bar:
        for step, batch in enumerate(train_loader):
            if step%10==0 and rank==0:
                print("Epoch:", epoch, ", step: ",step)
            
            # print("image.shape", batch["image"].shape)
            images = batch["image"].to(device)
            
            # reconstruction, z_mu, z_sigma = autoencoder(images)
            
            encoding_sample= autoencoder.encode_stage_2_inputs(images)
            print("encoding_sample", encoding_sample.shape)
            print("encoding_sample", encoding_sample[0, 0, ...].shape)
            for axis in range(3):
                        
                encoding_sample_img = visualize_one_slice_in_3d_image_greyscale(encoding_sample[0, 0, ...], axis) #.transpose([2, 1, 0])
                # encode = visualize_one_slice_in_3d_image_greyscale(encode[0, 0, ...], axis) #.transpose([2, 1, 0])
                # just_encoder = visualize_one_slice_in_3d_image_greyscale(just_encoder[0, 0, ...], axis) #.transpose([2, 1, 0])
                
                wandb.log({
                            f"train/encode_with_sampling_axis_{axis}": Image(encoding_sample_img),
                            # f"train/encode{axis}": Image(encode),
                            # f"train/encoder{axis}": Image(just_encoder),
                            
                            }, step=total_step*args.autoencoder_train["batch_size"]*world_size)
            del encoding_sample
            encode,_ = autoencoder.encode(images)
            print("encoding_sample", encode[0, 0, ...].shape)
            for axis in range(3):
                
                # encoding_sample = visualize_one_slice_in_3d_image_greyscale(encoding_sample[0, 0, ...], axis) #.transpose([2, 1, 0])
                encode_img = visualize_one_slice_in_3d_image_greyscale(encode[0, 0, ...], axis) #.transpose([2, 1, 0])
                # just_encoder = visualize_one_slice_in_3d_image_greyscale(just_encoder[0, 0, ...], axis) #.transpose([2, 1, 0])
                
                wandb.log({
                            # f"train/encode_with_sampling{axis}": Image(encoding_sample),
                            f"train/encode_axis_{axis}": Image(encode_img),
                            # f"train/encoder{axis}": Image(just_encoder),
                            
                            }, step=total_step*args.autoencoder_train["batch_size"]*world_size)
            del encode
            just_encoder = autoencoder.just_encoder(images)
            for axis in range(3):
                        
                # encoding_sample = visualize_one_slice_in_3d_image_greyscale(encoding_sample[0, 0, ...], axis) #.transpose([2, 1, 0])
                # encode = visualize_one_slice_in_3d_image_greyscale(encode[0, 0, ...], axis) #.transpose([2, 1, 0])
                just_encoder_img = visualize_one_slice_in_3d_image_greyscale(just_encoder[0, 0, ...], axis) #.transpose([2, 1, 0])
                
                wandb.log({
                            # f"train/encode_with_sampling{axis}": Image(encoding_sample),
                            # f"train/encode{axis}": Image(encode),
                            f"train/encoder_axis_{axis}": Image(just_encoder_img),
                            
                            }, step=total_step*args.autoencoder_train["batch_size"]*world_size)
            del just_encoder
            total_step += 1
        
        

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
    wandb.finish()

