import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism

from utils import define_instance, prepare_dataloader, setup_ddp, prepare_dataloader_extract_dataset_custom,  prepare_file_list

import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

# def convert_tensor_values(tensor):
#     mask_zero = tensor == 0
#     mask_one = tensor == 1

#     print("mask_zero", mask_zero)
#     print("mask_one", mask_one)
#     tensor_ori=tensor
#     # Convert 0s to 1s and 1s to 0s using the masks
#     tensor[mask_zero] = 1
#     tensor[mask_one] = 0
#     print("check equal",torch.equal(tensor_ori, tensor))
#     return tensor

import torch

def convert_tensor_values(tensor):
    # Assuming the tensor is on a CUDA device already
    mask_zero = tensor == 0
    mask_one = tensor == 1

    # print("mask_zero", mask_zero)
    # print("mask_one", mask_one)

    # Make a copy of the original tensor for comparison
    tensor_ori = tensor.clone()

    # Use temporary placeholders to avoid overwriting issues
    # It's crucial that the placeholder value does not exist in the tensor, hence -1 is chosen
    tensor[mask_zero] = -1  # Set 0s to -1 (a placeholder not in 0 or 1)
    tensor[mask_one] = 0    # Convert 1s to 0s
    tensor[tensor == -1] = 1  # Now convert placeholders (-1) to 1

    # Verify that the tensor has changed as expected
    # is_equal = torch.equal(tensor_ori, tensor)
    # print("Check equal:", is_equal)
    # print("tensor_ori, tensor", tensor_ori)
    # print("tensor_ori, tensor", tensor)

    return tensor, tensor_ori

# Example to test this function
# original_tensor = torch.tensor([[[0, 1, 2], [1, 0, 0], [1, 1, 1]]], device='cuda:0')
# print("Original Tensor:\n", original_tensor)

# converted_tensor = convert_tensor_values(original_tensor)
# print("Converted Tensor:\n", converted_tensor)


## Inversion
@torch.no_grad()
def invert(
    start_latents,
    original_condition,
    ddim_scheduler,
    diffusion_model,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=False,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    # text_embeddings = pipe._encode_prompt(
    #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    # )
    
    

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    # print("device",device)
    ddim_scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(ddim_scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):
        # print("inference step", i)
        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]
        t = t.unsqueeze(0)

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = ddim_scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        # noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # noise_pred = inferer(
        #                     inputs=latents ,
        #                     autoencoder_model=inferer_autoencoder,
        #                     diffusion_model=unet,
        #                     noise=noise,
        #                     timesteps=timesteps,
        #                     condition=condition,
        #                 )
        # print("t", t)
        noise_pred= diffusion_model(latent_model_input, timesteps=t, context=original_condition)
        



        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = ddim_scheduler.alphas_cumprod[current_t].to(device)
        # print("alpha_t_next device:", scheduler.alphas_cumprod.device)
        # print("next_t device:", next_t.device)
        # print(scheduler.device)
        ddim_scheduler.alphas_cumprod = ddim_scheduler.alphas_cumprod.to(device)
        alpha_t_next = ddim_scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Latent Diffusion Model Inference")
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
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="number of generated images",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print_config()
    # torch.backends.cudnn.benchmark = True
    # torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Set validation data loader
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
    
        _, val_loader = prepare_dataloader(
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
        
        _, val_loader =  prepare_dataloader_extract_dataset_custom(
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
        # print("args.diffusion_train[conditioning_file],", args.diffusion_train["conditioning_file"])
        _, val_loader =  prepare_dataloader_extract_dataset_custom(
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


    # load trained networks
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(args.autoencoder_dir, "autoencoder.pt")
    autoencoder.load_state_dict(torch.load(trained_g_path))

    diffusion_model = define_instance(args, "diffusion_def").to(device)
    trained_diffusion_path = os.path.join(args.diffusion_dir, "diffusion_unet.pt")
    diffusion_model.load_state_dict(torch.load(trained_diffusion_path))

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    latent_shape = [p // 4 for p in args.diffusion_train["patch_size"]]
    noise_shape = [1, args.latent_channels] + latent_shape




    # move from diffuser settings
    # Load a pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")#.to(device)
    # Set up a DDIM scheduler
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    del pipe
    # scheduler = DDIMScheduler.from_config(scheduler.config)
    # Plot 'alpha' (alpha_bar in DDPM language, alphas_cumprod in Diffusers for clarity)
    # timesteps = scheduler.timesteps.cpu()
    # alphas = scheduler.alphas_cumprod[timesteps]
    # plt.plot(timesteps, alphas, label="alpha_t")
    # plt.legend()



    # for _ in range(args.num):
    for step, batch in enumerate(val_loader):
        print("step", step)
        if step==1:
            break
        noise = torch.randn(noise_shape, dtype=torch.float32).to(device)
        print("noise", noise.shape)
        
        images = batch["image"].float().to(device)
        original_condition = batch["condition"].float().to(device)
        print("original_condition,", original_condition)
        latent_clean = autoencoder.encode_stage_2_inputs(images) * 1.0
        
        
        inverted_latents = invert(start_latents=latent_clean, 
                                  original_condition=original_condition, 
                                  ddim_scheduler=ddim_scheduler, 
                                  diffusion_model=diffusion_model,
                                  num_inference_steps=1000)
        
        # print("inverted_latents.shape",inverted_latents.shape)
        inverted_latents = inverted_latents[-1].unsqueeze(0)
        print("inverted_latents.shape",inverted_latents.shape)
        
        # converted_condition, tensor_ori= convert_tensor_values(original_condition)
        # print("original_condition ", torch.equal(original_condition, tensor_ori))
        # print("original_condition,", original_condition)
        # print("tensor ori,", tensor_ori)
        # print("converted_condition", converted_condition)
        # print("original_condition ", torch.equal(original_condition, converted_condition))
        # print("original_condition ", torch.equal(tensor_ori, converted_condition))
        
        
        # print("converted_condition", converted_condition.shape)
        np.set_printoptions(threshold=1000)
        # print("converted_condition", converted_condition[:,:,249])
        
        # print("converted_condition", converted_condition)
        print("original_condition", original_condition)
        
        
        
        
        with torch.no_grad():
            synthetic_images = inferer.sample(
                # input_noise=inverted_latents,
                input_noise=inverted_latents,
                autoencoder_model=autoencoder,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                conditioning=original_condition, 
                # conditioning=torch.tensor([[[138.,   1.,   0.]]]).to(device), 
                
            )
        # with torch.no_grad():
        #     synthetic_images = autoencoder.decode_stage_2_outputs(inverted_latents)
        filename = os.path.join(args.output_dir, datetime.now().strftime("original_%Y%m%d_%H%M%S"))
        final_img = nib.Nifti1Image(synthetic_images[0, 0, ...].unsqueeze(-1).cpu().numpy(), np.eye(4))
        nib.save(final_img, filename)


if __name__ == "__main__":# Load a pipeline

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()


