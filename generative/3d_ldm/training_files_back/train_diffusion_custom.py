import argparse
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm


import os
import sys

import torch
import torch.nn.functional as F
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import define_instance, prepare_dataloader, setup_ddp, prepare_dataloader_extract_dataset_custom,  prepare_file_list
from plot_test.visualize_image import visualize_one_slice_in_3d_image
from datetime import datetime
from create_dataset import *

import wandb
from wandb import Image

def main():
    parser = argparse.ArgumentParser(description="PyTorch Latent Diffusion Model Training")
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

    # print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
        
    from datetime import datetime
    # Generate a dynamic name based on the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'diffusion_{current_time}'
    if rank ==0:
        wandb.init(project=args.wandb_project_name_diffusion,name=run_name, config=args)

    set_determinism(42)

    # Step 1: set data loader
    # Choose base directory base on the cluster storage. 
    # set up environment variable accordingly e.g. "export CLUSTER_NAME=sc" 
    cluster_name = os.getenv('CLUSTER_NAME')
    if cluster_name == 'vassar':
        base_dir = '/home/sijun/meow/data_new/'
    elif cluster_name == 'sc':
        base_dir = '/scr/fangruih/mri_data/'
    else:
        raise ValueError('Unknown cluster name. Please set the CLUSTER_NAME environment variable. e.g. export CLUSTER=NAME=sc')

    size_divisible = 2 ** (len(args.diffusion_def["num_channels"]) - 1)
    # Step 1: set data loader
    file_list_cache = Path(f"{base_dir}/{args.dataset_type}_file_list.json")
    
    file_list_start_time = time.time()  # Record end time for data loader setup
    
    print(f"file_list_start_time: {file_list_start_time:.2f} seconds")
    if rank == 0:
        if file_list_cache.exists():
            with open(file_list_cache, 'r') as f:
                all_files_str = json.load(f)
                print("Loaded file list from cache.")
        else:
            file_list_start_time = time.time()  # Record end time for data loader setup
    
            print(f"file_list_start_time: {file_list_start_time:.2f} seconds")

            all_files = prepare_file_list(base_dir=base_dir, type=args.dataset_type)
            
            file_list_end_time = time.time()  # Record end time for data loader setup
    
            print(f"file_list_end_time: {file_list_end_time:.2f} seconds")
            # Convert Path objects to strings for JSON serialization
            all_files_str = [str(file) for file in all_files]
            dump_start_time = time.time()  # Record end time for data loader setup
    
            print(f"dump_start_time: {dump_start_time:.2f} seconds")

            with open(file_list_cache, 'w') as f:
                json.dump(all_files_str, f)
                print("Saved file list to cache.")
            
            dump_end_time = time.time()  # Record end time for data loader setup
    
            print(f"dump_end_time: {dump_end_time:.2f} seconds")

        # Convert the file list to a JSON string to broadcast
        all_files_json = json.dumps(all_files_str)
    else:
        all_files_json = None
    all_files_json_list = [all_files_json]
    if args.gpus > 1:
        dist.broadcast_object_list(all_files_json_list, src=0)
    all_files_json = all_files_json_list[0]

    # Convert the JSON string back to a list
    all_files_str = json.loads(all_files_json)
    all_files = [Path(file) for file in all_files_str]
    

    if args.dataset_type=="brain_tumor":
    
        train_loader, val_loader = prepare_dataloader(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
            amp=True,
        )
        
    elif args.dataset_type=="hcp_ya_T1":
        
        train_loader, val_loader =  prepare_dataloader_extract_dataset_custom(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            # base_dir= base_dir,
            all_files= all_files, 
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
            amp=True,
            with_conditioning=args.diffusion_def["with_conditioning"],
        )
    elif args.dataset_type=="T1_all":
        print("args.diffusion_train[conditioning_file],", args.diffusion_train["conditioning_file"])
        train_loader, val_loader =  prepare_dataloader_extract_dataset_custom(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            # base_dir= base_dir,
            all_files= all_files, 
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
            amp=True,
            with_conditioning=args.diffusion_def["with_conditioning"],
            conditioning_file=args.diffusion_train["conditioning_file"],
            cross_attention_dim=args.diffusion_def["cross_attention_dim"],
            expand_token_times= args.diffusion_train["expand_token_times"],
        )
        
    else: 
        raise ValueError(f"Unsupported dataset type specified: {args.dataset_type}")

    # initialize tensorboard writer and wandb
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # tensorboard_path = os.path.join(args.tfevent_path, "diffusion")
        tensorboard_path = os.path.join(args.tfevent_path, "diffusion", current_time)

        # Ensure the directory exists
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        
        
        
    # Step 2: Define Autoencoder KL network and diffusion model
    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    trained_g_path = os.path.join(args.autoencoder_dir, "autoencoder.pt")

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
    print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
            print("z.shape",z.shape)
            # z_2 = autoencoder.encode_stage_2_inputs_vae(check_data["image"].to(device))
            # print("z_2.shape",z_2.shape)
            if rank == 0:
                print(f"Latent feature shape {z.shape}")
                for axis in range(3):
                    train_img= visualize_one_slice_in_3d_image(check_data["image"][0, 0, ...], axis)#.transpose([2, 1, 0])
                    tensorboard_writer.add_image(
                        "train_img_" + str(axis),
                        # visualize_one_slice_in_3d_image(check_data["image"][0, 0, ...], axis).transpose([2, 1, 0]),
                        train_img.transpose([2, 1, 0]),
                        1,
                    )
                    
                    wandb.log({
                        f"val/image/gt_axis_{axis}": Image(train_img),
                        
                    }, step=1)
                print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    print(f"Rank {rank}: local scale_factor: {scale_factor}")
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {rank}: final scale_factor -> {scale_factor}")

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_def").to(device)
    args.model_dir= os.path.join(args.model_dir,current_time)
    # args.model_dir= os.path.join(args.diffusion_dir,current_time)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(args.model_dir, "diffusion_unet_last.pt")
    
    # trained_diffusion_path = os.path.join(args.diffusion_dir, "diffusion_unet.pt")
    # trained_diffusion_path_last = os.path.join(args.diffusion_dir, "diffusion_unet_last.pt")

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location))
            print(f"Rank {rank}: Load trained diffusion model from", trained_diffusion_path)
        except:
            print(f"Rank {rank}: Train diffusion model from scratch.")

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # Step 3: training config
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=args.diffusion_train["lr"] * world_size)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[100, 1000], gamma=0.1)

    # Step 4: training
    n_epochs = args.diffusion_train["n_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler()
    total_step = 0
    best_val_recon_epoch_loss = 100.0

    for epoch in range(n_epochs):
        unet.train()
        epoch_loss = 0
        lr_scheduler.step()
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        # train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{n_epochs}")
        
        # for step, batch in train_progress_bar:
        for step, batch in enumerate(train_loader):
            if step%10 == 0  and rank == 0:
                print("epoch:",epoch,", step:", step)
                
            images = batch["image"].to(device)
            
            if args.diffusion_def["with_conditioning"]:
                condition = batch["condition"].to(device)
                
                # print("condition.shape",condition.shape)
                # condition = torch.cat([item['condition'] for item in batch], dim=0).to(device)
            
            else: 
                condition=None
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder
                
                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=inferer_autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=condition,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            
            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss, total_step * args.diffusion_train["batch_size"]*world_size)
                wandb.log({
                    "train_diffusion_loss_iter": loss.item(),
                    }, step=total_step * args.diffusion_train["batch_size"]*world_size)
            # train_progress_bar.set_postfix(loss=loss.item())

        torch.cuda.empty_cache()
        # validation
        if epoch % val_interval == 0:
            
            autoencoder.eval()
            unet.eval()
            val_recon_epoch_loss = 0
            with torch.no_grad():
                with autocast(enabled=True):
                    # compute val loss
                    condition=None
                    for step, batch in enumerate(val_loader):
                        # if step>2:
                        #     break
                        images = batch["image"].to(device)
                        if args.diffusion_def["with_conditioning"]:
                            condition = batch["condition"].to(device)
                            # print("condition.shape",condition.shape)
                            # condition = torch.cat([item['condition'] for item in batch], dim=0).to(device)
            
                        else: 
                            condition = None
                        noise_shape = [images.shape[0]] + list(z.shape[1:])
                        noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                        
                        # print("list(z.shape[1:]", list(z.shape[1:]))
                        # print("images", images.shape)
                        # print()

                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        # Get model prediction
                        if ddp_bool:
                            inferer_autoencoder = autoencoder.module
                        else:
                            inferer_autoencoder = autoencoder
                        # print("images", images.shape)

                        noise_pred = inferer(
                            inputs=images,
                            autoencoder_model=inferer_autoencoder,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                            condition=condition,
                        )
                        
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                        val_recon_epoch_loss += val_loss
                    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

                    if ddp_bool:
                        dist.barrier()
                        dist.all_reduce(val_recon_epoch_loss, op=torch.distributed.ReduceOp.AVG)

                    val_recon_epoch_loss = val_recon_epoch_loss.item()

                    # write val loss and save best model
                    if rank == 0:
                        tensorboard_writer.add_scalar("val_diffusion_loss", val_recon_epoch_loss, epoch)
                        wandb.log({
                            "val_diffusion_loss": val_recon_epoch_loss,
                            }, step=total_step * args.diffusion_train["batch_size"]*world_size)
                        print(f"Epoch {epoch} val_diffusion_loss: {val_recon_epoch_loss}")
                        # save last model
                        if ddp_bool:
                            torch.save(unet.module.state_dict(), trained_diffusion_path_last)
                        else:
                            torch.save(unet.state_dict(), trained_diffusion_path_last)

                        # save best model
                        if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                            best_val_recon_epoch_loss = val_recon_epoch_loss
                            if ddp_bool:
                                torch.save(unet.module.state_dict(), trained_diffusion_path)
                            else:
                                torch.save(unet.state_dict(), trained_diffusion_path)
                            print("Got best val noise pred loss.")
                            print("Save trained latent diffusion model to", trained_diffusion_path)

                        # visualize synthesized image
                        if (epoch) % (2 * val_interval) == 0:  # time cost of synthesizing images is large
                            if condition!=None:
                                condition= condition[0].unsqueeze(0)
                                # print("syntesize base on condition ", condition)
                                print("condition shape", condition.shape)
                            
                            synthetic_images = inferer.sample(
                                input_noise=noise[0:1, ...],
                                autoencoder_model=inferer_autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                                conditioning=condition
                            )
                            for axis in range(3):
                                synthetic_img= visualize_one_slice_in_3d_image(synthetic_images[0, 0, ...], axis)
                                tensorboard_writer.add_image(
                                    "val_diff_synimg_" + str(axis),
                                    # visualize_one_slice_in_3d_image(synthetic_images[0, 0, ...], axis).transpose(
                                    #     [2, 1, 0]
                                    # ),
                                    synthetic_img.transpose([2, 1, 0]),
                                    epoch,
                                )
                                
                                wandb.log({
                                        f"val/image/syn_axis_{axis}": Image(synthetic_img),
                                        
                                    }, step=total_step*args.diffusion_train["batch_size"]*world_size)
                            
                            del synthetic_images
        torch.cuda.empty_cache()

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()