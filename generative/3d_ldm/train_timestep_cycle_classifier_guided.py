import argparse
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm


import os
import sys
import random

import torch
import torch.nn.functional as F
from generative.inferers import LatentDiffusionInferer
# from generative.networks.schedulers import DDPMScheduler
from generative.networks.schedulers import DDIMScheduler as monai_ddimscheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.amp import GradScaler
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Flatten
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler

from utils import define_instance, prepare_dataloader, setup_ddp, prepare_dataloader_extract_dataset_custom,  prepare_file_list, count_parameters
from util.dataset_utils import prepare_dataloader_from_list
from util.training_utils import invert, convert_tensor_age, print_gpu_usage
from plot_test.visualize_image import visualize_one_slice_in_3d_image
from datetime import datetime
from create_dataset import *
from util.resnet3D import resnet50
from util.image_conversion_utils import conversion
from util.image_inference_utils import generate_image_from_condition
from torch.nn.utils import parameters_to_vector, vector_to_parameters


import wandb
from wandb import Image

from monai.networks.nets import Regressor


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
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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
    run_name = f'cycle_age{args.cycle_param["age_loss_weight"]}_cycle{args.cycle_param["cycle_loss_weight"]}_transfer{args.cycle_param["transfer_loss_weight"]}_penalty{args.cycle_param["weight_penalty_weight"]}_cycletransfer{args.cycle_param["cycle_transfer_weight"]}'
    
    if rank ==0:
        wandb.init(project=args.wandb_project_name_cycle,name=run_name, config=args)
        wandb.config.update({"current_time": current_time})

    set_determinism(42)

    # Step 1: set data loader
    # Choose base directory base on the cluster storage. 
    # set up environment variable accordingly e.g. "export CLUSTER_NAME=sc" 
    cluster_name = os.getenv('CLUSTER_NAME')
    if cluster_name == 'vassar':
        base_dir = '/home/sijun/meow/data_new/'
    elif cluster_name == 'sc':
        base_dir = '/scr/fangruih/mri_data/'
    elif cluster_name == 'haic':
        base_dir = '/hai/scratch/fangruih/data/'
    else:
        raise ValueError(f'Unknown cluster name{cluster_name}. Please set the CLUSTER_NAME environment variable. e.g. export CLUSTER=NAME=sc')

    size_divisible = 2 ** (len(args.diffusion_def["num_channels"]) - 1)

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
    
    elif args.dataset_type=="T1_all":
        train_loader, val_loader =  prepare_dataloader_from_list(
            args,
            args.diffusion_train["batch_size"],
            args.diffusion_train["patch_size"],
            randcrop=False,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            size_divisible=size_divisible,
            amp=True,
            with_conditioning=args.diffusion_def["with_conditioning"],
            
            cross_attention_dim=args.diffusion_def["cross_attention_dim"],
            expand_token_times= args.diffusion_train["expand_token_times"],
        )
        
    else: 
        raise ValueError(f"Unsupported dataset type specified: {args.dataset_type}")

    # initialize tensorboard writer and wandb
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
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

    
    # age regressor model 
    # age_regressor = Regressor(in_shape=[1,148,180,148], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)).to(device)
    age_regressor = resnet50(shortcut_type='B').to(device)
    # Modify the model structure
    age_regressor.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        Flatten(),
        nn.Linear(2048, 1),  # Final output layer for age prediction
        nn.ReLU()
    ).to(device)
    regressor_state_dict = torch.load('/hai/scratch/fangruih/monai-tutorials/3d_regression/trained_model/resnet/best_metric_model_20241003_003900.pth')


    # Load the state dictionary into the model
    age_regressor.load_state_dict(regressor_state_dict)
        
    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
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
    args.cycle_model_dir= os.path.join(args.cycle_model_dir,current_time)
    # args.model_dir= os.path.join(args.diffusion_dir,current_time)
    Path(args.cycle_model_dir).mkdir(parents=True, exist_ok=True)
    # print(unet)
    
    trained_diffusion_path = os.path.join(args.diffusion_dir, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(args.diffusion_dir, "diffusion_unet_last.pt")
    
    # trained_diffusion_path = os.path.join(args.diffusion_dir, "diffusion_unet.pt")
    # trained_diffusion_path_last = os.path.join(args.diffusion_dir, "diffusion_unet_last.pt")

    cycle_diffusion_path = os.path.join(args.cycle_model_dir, "diffusion_unet.pt")
    cycle_diffusion_path_last = os.path.join(args.cycle_model_dir, "diffusion_unet_last.pt")

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location))
            print(f"Rank {rank}: Load trained diffusion model from", trained_diffusion_path)
        except:
            print(f"Rank {rank}: Train diffusion model from scratch.")

    # scheduler = DDPMScheduler(
    #     num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
    #     schedule="scaled_linear_beta",
    #     beta_start=args.NoiseScheduler["beta_start"],
    #     beta_end=args.NoiseScheduler["beta_end"],
    # )
    scheduler = monai_ddimscheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        # schedule="scaled_linear_beta",
        # beta_start=args.NoiseScheduler["beta_start"],
        # beta_end=args.NoiseScheduler["beta_end"],
    )
    scheduler.set_timesteps(50)
    num_params = count_parameters(unet)
    print(f'The model has {num_params} trainable parameters.')
    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)
        age_regressor = DDP(age_regressor, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # move from diffuser settings
    # Load a pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")#.to(device)
    # Set up a DDIM scheduler
    diffuser_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    del pipe
    torch.cuda.empty_cache()
    
    if rank == 0:
        initial_weights = parameters_to_vector(unet.parameters()).detach()
        
    
    # Step 3: training config
    # optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=args.diffusion_train["lr"] * world_size)
    
    optimizer_diff = torch.optim.AdamW(
        unet.parameters(),
        lr= args.diffusion_train["lr"],
        betas= (0.9, 0.999),
        weight_decay= 1e-2,
        eps= 1e-08,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[100, 1000], gamma=0.1)

    # Step 4: training
    n_epochs = args.diffusion_train["n_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    age_regressor.eval()
    scaler = GradScaler()
    total_step = 0
    best_val_recon_epoch_loss = 100.0

    # Create a directory for checkpoints
    checkpoint_dir = os.path.join(args.cycle_model_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

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
            # if step>2:
            #     break
            torch.cuda.empty_cache()
            if step%10 == 0  and rank == 0:
                print("epoch:",epoch,", step:", step)
                
            images = batch["image"].to(device)
            if args.diffusion_def["with_conditioning"]:
                condition_x = batch["condition"].to(device)
                if step<1:
                    print("condition.shape",condition_x.shape)
                    print("condition.shape",condition_x)
            else: 
                condition_x = None
                
            
            optimizer_diff.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            with autocast(device_type='cuda', enabled=True):
                torch.cuda.empty_cache()
                        
                
                
                # Step 1 : encode the original image to get clean latent 
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder
                with torch.no_grad():
                    x_0 = inferer_autoencoder.encode_stage_2_inputs(images) * scale_factor
                torch.cuda.empty_cache()
                
                
                # Step 2: sample random noise x
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                noise_x = torch.randn(noise_shape, dtype=images.dtype).to(device)
                
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()
                
                
                # Step 3: Add noise to x_0 get x_t
                x_t = inferer.scheduler.add_noise(original_samples=x_0, noise=noise_x, timesteps=timesteps)
                
                # Step 4: get a different condition 
                age_y = random.uniform(0, 90)
                condition_y = convert_tensor_age(condition_x,age=age_y)
                condition_80 = convert_tensor_age(condition_x,age=80)
                condition_10 = convert_tensor_age(condition_x,age=10)
                
                # Step 5: predict noise and compare it to noise_x
                if step % 2 == 0:
                    predicted_noise_x = unet(x=x_t, timesteps=timesteps, context=condition_y)
                else:
                    with torch.no_grad():
                        predicted_noise_x = unet(x=x_t, timesteps=timesteps, context=condition_y)
                # predicted_noise_x = unet(x=x_t, timesteps=timesteps, context=condition_y)
                
                # Step 6: get fake y_0
                
                y_0_fake = inferer.scheduler.reversed_step_t0(model_output=predicted_noise_x, timestep=timesteps, sample=x_t)
                
                # Step 7: Sample random noise y
                noise_y = torch.randn(noise_shape, device=device).to(images.dtype)
                # noise_y = torch.randn(noise_shape, dtype=images.dtype).to(device)

                
                # Step 8: 
                y_t = inferer.scheduler.add_noise(original_samples=y_0_fake, noise=noise_y, timesteps=timesteps)
                
                # Step 9: predict noise and compare it to noise_y
                if step % 2 == 0:
                    with torch.no_grad():
                        predicted_noise_y = unet(x=y_t, timesteps=timesteps, context=condition_x)
                else:
                    predicted_noise_y = unet(x=y_t, timesteps=timesteps, context=condition_x)
                # predicted_noise_y = unet(x=y_t, timesteps=timesteps, context=condition_x)
                
                # Step 11: predict age
                torch.cuda.empty_cache()
                with torch.no_grad():
                    y_0_fake_image = inferer_autoencoder.decode_stage_2_outputs(y_0_fake)   

                predicted_age = age_regressor(y_0_fake_image)
                
                age_y = torch.tensor(age_y, dtype=torch.float32, device=predicted_age.device).view_as(predicted_age)
                
                # Step 12: predict x_0_fake
                
                x_0_fake = inferer.scheduler.reversed_step_t0(model_output=predicted_noise_y, timestep=timesteps, sample=y_t)
                
                # Steps 12: calculate losses
                if step % 2 == 0:
                    transfer_loss = F.mse_loss(predicted_noise_y.float(), noise_y.float()) + F.mse_loss(predicted_noise_x.float(), noise_x.float())
                else:
                    transfer_loss = F.mse_loss(predicted_noise_x.float(), noise_x.float()) + F.mse_loss(predicted_noise_y.float(), noise_y.float())
                age_loss = F.mse_loss(predicted_age, age_y)
                cycle_loss = F.mse_loss(x_0, x_0_fake)
                cycle_transfer_loss = F.mse_loss(predicted_noise_y.float() + predicted_noise_x.float() - noise_y.float() - noise_x.float(), torch.zeros_like(noise_y.float()))
                
                # Calculate weight change penalty
                current_weights = parameters_to_vector(unet.parameters())
                weight_change = F.mse_loss(current_weights, initial_weights)
                
                # Add weight change penalty to the loss
                weight_penalty_factor = 0.01  # Adjust this value to control the strength of the penalty
                weight_penalty_loss = weight_penalty_factor * weight_change

                
                # Scale the losses by their respective weights and add them together
                loss = args.cycle_param["transfer_loss_weight"]*transfer_loss + \
                       args.cycle_param["age_loss_weight"]*age_loss + \
                       args.cycle_param["cycle_loss_weight"]*cycle_loss + \
                       args.cycle_param["weight_penalty_weight"]*weight_penalty_loss + \
                       args.cycle_param["cycle_transfer_weight"]*cycle_transfer_loss
                
                
                
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            
            if args.diffusion_train["gradient_clipping"] :
                # scaler.unscale_(optimizer_diff)  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=args.diffusion_train["max_norm"])

            scaler.step(optimizer_diff)
            scaler.update()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss, total_step * args.diffusion_train["batch_size"]*world_size)
                wandb.log({
                    "train_transfer_loss_iter": transfer_loss.item(),
                    "train_age_loss_iter": age_loss.item(),
                    "train_cycle_loss_iter": cycle_loss.item(),
                    "train_cycle_transfer_loss_iter": cycle_transfer_loss.item(),
                    "train_total_loss_iter": loss.item(),
                }, step=total_step * args.diffusion_train["batch_size"]*world_size)
                
            # Store checkpoint every 1000 iterations
            if total_step % 10000 == 0 and rank == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{total_step}.pt")
                checkpoint = {
                    'epoch': epoch,
                    'step': total_step,
                    'model_state_dict': unet.module.state_dict() if ddp_bool else unet.state_dict(),
                    'optimizer_state_dict': optimizer_diff.state_dict(),
                    "train_transfer_loss_iter": transfer_loss.item(),
                    "train_age_loss_iter": age_loss.item(),
                    "train_cycle_loss_iter": cycle_loss.item(),
                    "train_cycle_transfer_loss_iter": cycle_transfer_loss.item(),
                    "train_weight_loss_iter": weight_penalty_loss.item(),
                    "train_total_loss_iter": loss.item(),
                    
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at iteration {total_step} to {checkpoint_path}")

            
        
            # After the existing training step
            if step % 101 == 0 and rank == 0:
                unet.eval()
                with torch.no_grad():
                    # Generate image from condition
                    print("noise_shape",noise_shape)
                    generated_image_80 = generate_image_from_condition(
                        condition_80,
                        autoencoder,
                        unet,
                        scheduler,
                        inferer,
                        device,
                        noise_shape,
                        save_path=None
                    )
                    generated_image_10 = generate_image_from_condition(
                        condition_10,
                        autoencoder,
                        unet,
                        scheduler,
                        inferer,
                        device,
                        noise_shape,
                        save_path=None
                    )


                    
                    # Convert image to old age 
                    converted_image_80, _, original_image, _ = conversion(
                        images,
                        condition_x,
                        condition_80,
                        autoencoder,
                        unet,
                        diffuser_scheduler,
                        scheduler,
                        inferer,
                        device,
                        save_path=None
                    )
                    # Convert image to young age 
                    converted_image_10, _, original_image, _ = conversion(
                        images,
                        condition_x,
                        condition_10,
                        autoencoder,
                        unet,
                        diffuser_scheduler,
                        scheduler,
                        inferer,
                        device,
                        save_path=None
                    )
                    # Convert image to young age 
                    converted_image_y, _, original_image, _ = conversion(
                        images,
                        condition_x,
                        condition_y,
                        autoencoder,
                        unet,
                        diffuser_scheduler,
                        scheduler,
                        inferer,
                        device,
                        save_path=None
                    )
                    # Log images and conditions
                    original_age, original_sex = condition_x[0, 0, 0].item(), condition_x[0, 0, 1].item()
                    new_age, new_sex = condition_y[0, 0, 0].item(), condition_y[0, 0, 1].item()

                    for axis in range(3):
                        original_img = visualize_one_slice_in_3d_image(original_image[0, 0, ...], axis)
                        generated_img_80 = visualize_one_slice_in_3d_image(generated_image_80[0, 0, ...], axis)
                        generated_img_10 = visualize_one_slice_in_3d_image(generated_image_10[0, 0, ...], axis)
                        converted_img_80 = visualize_one_slice_in_3d_image(converted_image_80[0, 0, ...], axis)
                        converted_img_10 = visualize_one_slice_in_3d_image(converted_image_10[0, 0, ...], axis)
                        converted_img_y = visualize_one_slice_in_3d_image(converted_image_y[0, 0, ...], axis)
                        img_y_0 = visualize_one_slice_in_3d_image(y_0_fake_image[0, 0, ...], axis)
                        img_x_0 = visualize_one_slice_in_3d_image(images[0, 0, ...], axis)
                        
                        
                        wandb.log({
                            f"train_original_image/original_axis_{axis}": wandb.Image(original_img),
                            f"train_generation_80/generated_axis_{axis}": wandb.Image(generated_img_80),   
                            f"train_generation_10/generated_axis_{axis}": wandb.Image(generated_img_10),
                            
                            f"train_conversion_80/converted_80_axis_{axis}": wandb.Image(converted_img_80),
                            f"train_conversion_10/converted_10_axis_{axis}": wandb.Image(converted_img_10),
                            f"train_conversion_y/converted_y_axis_{axis}": wandb.Image(converted_img_y),
                            
                            f"train_intermediate/fake_y0_{axis}": wandb.Image(img_y_0),
                            f"train_intermediate/real_x0_{axis}": wandb.Image(img_x_0),
                            "train_conversion/original_condition/age": original_age,
                            "train_conversion/original_condition/sex": original_sex,
                            "train_conversion/condition_y/age": new_age,
                            "train_conversion/condition_y/sex": new_sex,
                        }, step=total_step * args.diffusion_train["batch_size"] * world_size)

                unet.train()


            del images, condition_x, condition_y, condition_80, condition_10, loss
            del x_0, x_t, y_t, y_0_fake, noise_x, noise_y, predicted_noise_x, predicted_noise_y, x_0_fake, predicted_age
            torch.cuda.empty_cache()
            
        # validation
        if epoch % val_interval == 0:
            
            autoencoder.eval()
            age_regressor.eval()
            unet.eval()
            val_recon_epoch_loss = 0
            with torch.no_grad():
                with autocast(device_type='cuda', enabled=True):
                    # compute val loss
                    condition_x=None
                    for step, batch in enumerate(val_loader):
                        torch.cuda.empty_cache()
                        # if step>2:
                        #     break
                        images = batch["image"].to(device)
                        if args.diffusion_def["with_conditioning"]:
                            condition_x = batch["condition"].to(device)
                            if step<2:
                                print("validation condition.shape",condition_x.shape)
                                print("validation condition.shape",condition_x)
                        else: 
                            condition_x = None
                        

                        # Step 1: encode the original image to get clean latent 
                        if ddp_bool:
                            inferer_autoencoder = autoencoder.module
                        else:
                            inferer_autoencoder = autoencoder
                        with torch.no_grad():
                            x_0 = inferer_autoencoder.encode_stage_2_inputs(images) * scale_factor
                        torch.cuda.empty_cache()
                
                
                        # Step 2: sample random noise x
                        noise_shape = [images.shape[0]] + list(z.shape[1:])
                        noise_x = torch.randn(noise_shape, dtype=images.dtype).to(device)
                        
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()
                        
                        
                        # Step 3: Add noise to x_0 get x_t
                        x_t = inferer.scheduler.add_noise(original_samples=x_0, noise=noise_x, timesteps=timesteps)
                        
                        # Step 4: get a different condition 
                        random_age = random.uniform(0, 90)
                        condition_y = convert_tensor_age(condition_x, age=random_age)
                        
                        # Step 5: predict noise and compare it to noise_x
                        predicted_noise_x = unet(x=x_t, timesteps=timesteps, context=condition_y)
                        
                        # Step 6: get fake y_0
                        y_0_fake = inferer.scheduler.reversed_step_t0(model_output=predicted_noise_x, timestep=timesteps, sample=x_t)
                        
                        # Step 7: Sample random noise y
                        noise_y = torch.randn(noise_shape, device=device).to(images.dtype)
                        # noise_y = torch.randn(noise_shape, dtype=images.dtype).to(device)

                        
                        # Step 8: 
                        y_t = inferer.scheduler.add_noise(original_samples=y_0_fake, noise=noise_y, timesteps=timesteps)
                        
                        # Step 9: predict noise and compare it to noise_y
                        predicted_noise_y = unet(x=y_t, timesteps=timesteps, context=condition_x)
                        
                        # Step 10: predict x_0_fake
                        x_0_fake = inferer.scheduler.reversed_step_t0(model_output=predicted_noise_y, timestep=timesteps, sample=y_t)
                        
                        # Steps 12: calculate losses
                        if step % 2 == 0:
                            transfer_loss = F.mse_loss(predicted_noise_y.float(), noise_y.float()) + F.mse_loss(predicted_noise_x.float(), noise_x.float())
                        else:
                            transfer_loss = F.mse_loss(predicted_noise_x.float(), noise_x.float()) + F.mse_loss(predicted_noise_y.float(), noise_y.float())
                        age_loss = F.mse_loss(predicted_age, age_y)
                        cycle_loss = F.mse_loss(x_0, x_0_fake)
                        
                        # Scale the losses by their respective weights  and add them together
                        val_loss = args.cycle_param["transfer_loss_weight"]*transfer_loss + args.cycle_param["age_loss_weight"]*age_loss + args.cycle_param["cycle_loss_weight"]*cycle_loss
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
                            "val_transfer_loss": transfer_loss.item(),
                            "val_age_loss": age_loss.item(),
                            "val_cycle_loss": cycle_loss.item(),
                            "val_total_loss": val_loss.item(),
                            "val_diffusion_loss": val_recon_epoch_loss,
                            }, step=total_step * args.diffusion_train["batch_size"]*world_size)
                        print(f"Epoch {epoch} val_diffusion_loss: {val_recon_epoch_loss}")
                        # save last model
                        if ddp_bool:
                            torch.save(unet.module.state_dict(), cycle_diffusion_path_last)
                        else:
                            torch.save(unet.state_dict(), cycle_diffusion_path_last)

                        # save best model
                        if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                            best_val_recon_epoch_loss = val_recon_epoch_loss
                            if ddp_bool:
                                torch.save(unet.module.state_dict(), cycle_diffusion_path)
                            else:
                                torch.save(unet.state_dict(), cycle_diffusion_path)
                            print("Got best val noise pred loss.")
                            print("Save trained latent diffusion model to", cycle_diffusion_path)

                        # visualize synthesized image
                        if (epoch) % (2 * val_interval) == 0:  # time cost of synthesizing images is large
                            if condition_x!=None:
                                condition_x= condition_x[0].unsqueeze(0)
                                
                            synthetic_images = inferer.sample(
                                # input_noise=noise_x,
                                input_noise=noise_x[0:1, ...],
                                autoencoder_model=inferer_autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                                conditioning=condition_x
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
        torch.cuda.empty_cache()

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
