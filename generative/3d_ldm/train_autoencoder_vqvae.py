
import argparse
import json
import logging

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

from utils import KL_loss, define_instance, prepare_dataloader, setup_ddp, print_gpu_memory
from utils import KL_loss, define_instance, prepare_dataloader, prepare_dataloader_extract_dataset, prepare_dataloader_extract_dataset_custom, setup_ddp, print_gpu_memory
from visualize_image import visualize_one_slice_in_3d_image, visualize_one_slice_in_3d_image_greyscale
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
        
    from datetime import datetime
    # Generate a dynamic name based on the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'vq_vae_{current_time}'
    wandb.init(project=args.wandb_project_name,name=run_name, config=args)

    set_determinism(42)

    # Step 1: set data loader
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
        base_dir = '/home/sijun/meow/data_new/hcp/registered'
        base_path = Path(base_dir)
        # all_files = list(base_path.rglob('*/**/3T/T1w_MPR1/*_3T_T1w_MPR1.nii.gz'))
        all_files = list(base_path.rglob('*/3T/T1w_MPR1/*_3T_T1w_MPR1.nii.gz'))

        # train_dataset, val_dataset = create_train_val_datasets(base_dir )
        train_loader, val_loader =  prepare_dataloader_extract_dataset_custom(
            args,
            args.autoencoder_train["batch_size"],
            args.autoencoder_train["patch_size"],
            # base_dir= base_dir,
            all_files= all_files, 
            randcrop=True,
            rank=rank,
            world_size=world_size,
            cache=1.0,
            download=False,
            size_divisible=size_divisible,
            amp=False,
        )
    else: 
        raise ValueError(f"Unsupported dataset type specified: {args.dataset_type}")


    # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    def count_model_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("count_model_parameters(model)", count_model_parameters(autoencoder))
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
    trained_g_path = os.path.join(args.model_dir,"vq_vae.pt")
    trained_d_path = os.path.join(args.model_dir, "vq_vae_discriminator.pt")
    trained_g_path_last = os.path.join(args.model_dir,"vq_vae_last.pt")
    trained_d_path_last = os.path.join(args.model_dir,"vq_vae_discriminator_last.pt")
    # print("trained_d_path_last", trained_d_path_last)

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
    quantization_weight = args.autoencoder_train["quantization_weight"]

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
        
        train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{n_epochs}")
    
        for step, batch in train_progress_bar:
            
            # print("image.shape", batch["image"].shape)
            images = batch["image"].to(device)
            # images = batch.to(device)
            # print("images", images.shape)
            
            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            
            # reconstruction, z_mu, z_sigma = autoencoder(images)
            # recons_loss = intensity_loss(reconstruction, images)
            # kl_loss = KL_loss(z_mu, z_sigma)
            # p_loss = loss_perceptual(reconstruction.float(), images.float())
            # loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss
            # Forward pass
            reconstruction, quantized, quantization_loss, z = autoencoder(images)
            
            # Compute losses
            recons_loss = intensity_loss(reconstruction, images)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            
            # Combine all losses
            loss_g = recons_loss + perceptual_weight * p_loss + quantization_weight * quantization_loss
            
            

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            # update progress bar
            train_progress_bar.set_postfix({
                'Loss': loss_g.item(),
                'Recon Loss': recons_loss.item(),
                'Quantization Loss': quantization_loss.item()
            })
            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                
                train_metrics = {
                    "train/recon_loss_iter": recons_loss.item(),  # Ensure to use .item() to log scalar values
                    "train/quantization_loss_iter": quantization_loss.item(),
                    "train/perceptual_loss_iter": p_loss.item()
                }
                tensorboard_writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)
                tensorboard_writer.add_scalar("train_quantization_loss_iter", quantization_loss, total_step)
                tensorboard_writer.add_scalar("train_perceptual_loss_iter", p_loss, total_step)
                if epoch > autoencoder_warm_up_n_epochs:
                    train_metrics.update({
                        "train/adv_loss_iter": generator_loss.item(),
                        "train/fake_loss_iter": loss_d_fake.item(),
                        "train/real_loss_iter": loss_d_real.item()
                    })
                    tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
                    tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
                    tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)
                wandb.log( train_metrics, step=total_step* args.autoencoder_train["batch_size"])

        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            mses= 0
            psnrs=0
            ssims=0
            mmd =0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)  # choose only one of Brats channels
                with torch.no_grad():
                    # reconstruction, z_mu, z_sigma = autoencoder(images)
                    reconstruction, quantized, quantization_loss, z = autoencoder(images)
                    # print("reconstruction", reconstruction.size())
                    # print("images", images.size())
                    
                    recons_loss = intensity_loss(
                        reconstruction.float(), images.float()
                    ) + perceptual_weight * loss_perceptual(reconstruction.float(), images.float())

                # mses_add,psnrs_add,ssims_add,mmd_add = metrics_mean_mses_psnrs_ssims_mmd(reconstruction,images)
                mses_add,psnrs_add,ssims_add = metrics_mean_mses_psnrs_ssims_mmd(reconstruction,images)
                
                mses= mses+mses_add
                psnrs= psnrs+psnrs_add
                ssims= ssims+ssims_add
                # mmd= mmd+mmd_add
                val_recon_epoch_loss += recons_loss.item()

            mses= mses/(step+1)
            psnrs= psnrs/(step+1)
            ssims= ssims/(step+1)
            # mmd= mmd/(step+1)
            
            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
            if rank == 0:
                # save last model
                print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")
                
                
                if ddp_bool:
                    torch.save(autoencoder.module.state_dict(), trained_g_path_last)
                    torch.save(discriminator.module.state_dict(), trained_d_path_last)
                else:
                    torch.save(autoencoder.state_dict(), trained_g_path_last)
                    torch.save(discriminator.state_dict(), trained_d_path_last)
                # save best model
                if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                    best_val_recon_epoch_loss = val_recon_epoch_loss
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path)
                        torch.save(discriminator.module.state_dict(), trained_d_path)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path)
                        torch.save(discriminator.state_dict(), trained_d_path)
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)

                # write val loss for each epoch into tensorboard
                val_metrics = {
                    "val/recon_loss": val_recon_epoch_loss,  # Ensure to use .item() to log scalar values
                    "val/mses": mses,
                    "val/psnrs": psnrs,
                    "val/ssims": ssims,
                    # "val/mmd": mmd,
                }
                # wandb.log(log_dict, step=total_step)
                tensorboard_writer.add_scalar("val_recon_loss", val_recon_epoch_loss, epoch)
                wandb.log(val_metrics, step=total_step * args.autoencoder_train["batch_size"])

                for axis in range(3):
                    tensorboard_writer.add_image(
                        "val_img_" + str(axis),
                        visualize_one_slice_in_3d_image(images[0, 0, ...], axis).transpose([2, 1, 0]),
                        epoch,
                    )
                    tensorboard_writer.add_image(
                        "val_recon_" + str(axis),
                        visualize_one_slice_in_3d_image(reconstruction[0, 0, ...], axis).transpose([2, 1, 0]),
                        epoch,
                    )
                    
                    val_img = visualize_one_slice_in_3d_image_greyscale(images[0, 0, ...], axis) #.transpose([2, 1, 0])
                    val_recon = visualize_one_slice_in_3d_image_greyscale(reconstruction[0, 0, ...], axis) #.transpose([2, 1, 0])
                    
                    # print("images.shape",images.shape)
                    # print("val_img.shape",val_img.shape)
                    # print("val_recon.shape",val_recon.shape)

                    wandb.log({
                        f"val/image/img_axis_{axis}": Image(val_img),
                        f"val/image/recon_axis_{axis}": Image(val_recon)
                    }, step=total_step*args.autoencoder_train["batch_size"])
                

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
    wandb.finish()

