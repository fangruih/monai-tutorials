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
from generative.networks.schedulers import DDPMScheduler as monai_ddpmscheduler
from generative.networks.schedulers import DDIMScheduler as monai_ddimscheduler

from monai.config import print_config
from monai.utils import set_determinism

from util.dataset_utils import prepare_dataloader_from_list, extract_age_sex_verify, create_converted_sex_tensor
from util.training_utils import print_gpu_usage
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

def find_unique_non_binary_number(tensor):
    # Flatten the tensor to make it easier to handle
    flat_tensor = tensor.flatten()

    # Get the unique elements in the tensor
    unique_elements = torch.unique(flat_tensor)

    # Filter out the common elements (0 and 1)
    # Since you mentioned there's only one number that's not 0 or 1, we directly find it
    non_binary_number = [x.item() for x in unique_elements if x not in (0, 1)]

    # Check if we found exactly one unique number
    if len(non_binary_number) == 1:
        return non_binary_number[0]
    else:
        raise ValueError("Expected exactly one unique number that is not 0 or 1, but found multiple or none.")



def convert_tensor_sex(tensor):
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


def convert_tensor_age(tensor,age):
    # Clone the tensor to avoid modifying the original
    new_tensor = tensor.clone()
    
    # Mask to identify elements that are not 0 or 1
    mask = (new_tensor != 0) & (new_tensor != 1)
    
    # Apply the mask to set the desired elements to 10
    new_tensor[mask] = age
    
    return new_tensor
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

        noise_pred= diffusion_model(latent_model_input, timesteps=t, context=original_condition)
        

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = ddim_scheduler.alphas_cumprod[current_t].to(device)
        
        ddim_scheduler.alphas_cumprod = ddim_scheduler.alphas_cumprod.to(device)
        alpha_t_next = ddim_scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

def main(age_or_sex):
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
        _, val_loader =  prepare_dataloader_from_list(
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


    # load trained networks
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(args.autoencoder_dir, "autoencoder.pt")
    autoencoder.load_state_dict(torch.load(trained_g_path))

    diffusion_model = define_instance(args, "diffusion_def").to(device)
    trained_diffusion_path = os.path.join(args.diffusion_dir, "diffusion_unet.pt")
    diffusion_model.load_state_dict(torch.load(trained_diffusion_path))

    # scheduler = monai_ddpmscheduler(
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
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    latent_shape = [p // 4 for p in args.diffusion_train["patch_size"]]
    noise_shape = [1, args.latent_channels] + latent_shape




    # move from diffuser settings
    # Load a pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")#.to(device)
    # Set up a DDIM scheduler
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    del pipe
    
    
    common_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if age_or_sex=="age":
        subfolder_path = os.path.join(args.output_dir, "counterfactual","age",common_timestamp)
    else: 
        subfolder_path = os.path.join(args.output_dir, "counterfactual","sex",common_timestamp)
    # subfolder_path = os.path.join(args.output_dir, "age","test")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        
    # for _ in range(args.num):
    for step, batch in enumerate(val_loader):
        torch.cuda.empty_cache()
        print("step", step)
        if step>args.num:
            break
        # if step==0 or step==1 or step==2:
        #     continue
        noise = torch.randn(noise_shape, dtype=torch.float32).to(device)
        print("noise", noise.shape)
        
        images = batch["image"].float().to(device)
        original_condition = batch["condition"].float().to(device)
        del batch
        print_gpu_usage()
        torch.cuda.empty_cache()
        latent_clean = autoencoder.encode_stage_2_inputs(images) * 1.0
        print_gpu_usage()
        
        print("before invert")
        print_gpu_usage()
        torch.cuda.empty_cache()
        inverted_latents = invert(start_latents=latent_clean, 
                                  original_condition=original_condition, 
                                  ddim_scheduler=ddim_scheduler, 
                                  diffusion_model=diffusion_model,
                                  num_inference_steps=50)
        print_gpu_usage()
        del latent_clean
        
        inverted_latents = inverted_latents[-10].unsqueeze(0)
        print_gpu_usage()
        torch.cuda.empty_cache()
        # converted_condition, tensor_ori = convert_tensor_values(original_condition)
        np.set_printoptions(threshold=1000)
       
        
        ori_age = int(find_unique_non_binary_number(original_condition))
        
        ori_filename = os.path.join(subfolder_path, f"index_{step}_oriAge_{ori_age}_ori")
        ori_img = nib.Nifti1Image(images[0, 0, ...].unsqueeze(-1).cpu().numpy(), np.eye(4))
        nib.save(ori_img, ori_filename)
        del ori_img
        
        
        # for age in ages:
        if age_or_sex=="age":
            age, sex= extract_age_sex_verify(original_condition)
            new_age= torch.randint(0, 91, (1,), dtype=torch.float32)
            converted_condition= convert_tensor_age(original_condition, age = new_age)
            print("new", new_age)
           
            filename = os.path.join(subfolder_path, f"index_{step}_oriAge_{ori_age}_newAge_{int(new_age)}")
            
        else:
            age, sex= extract_age_sex_verify(original_condition)
            new_sex = 1- sex
            converted_condition= create_converted_sex_tensor(original_condition)
            filename = os.path.join(subfolder_path, f"index_{step}_oriSex_{sex}_newSex_{new_sex}")
        #     converted_condition= convert_tensor_sex(original_condition)
    
        print("original_condition", original_condition)
        print("converted_condition", converted_condition)
        # age_converted_condition = convert_tensor_age(original_condition, age=age )
        
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Device for images:", images.device)
            print("Device for autoencoder model:", next(autoencoder.parameters()).device)
            print("Device for UNet diffusion model:", next(diffusion_model.parameters()).device)
            
            torch.cuda.empty_cache()
            synthetic_images = inferer.sample(
                input_noise=inverted_latents,
                autoencoder_model=autoencoder,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                conditioning=converted_condition, 
                
            )
            print("after sampling")
            print_gpu_usage()
        # with torch.no_grad():
        #     synthetic_images = autoencoder.decode_stage_2_outputs(inverted_latents)
        # filename = os.path.join(subfolder_path, f"index_{step}_oriAge_{ori_age}_newAge_{age}")
        final_img = nib.Nifti1Image(synthetic_images[0, 0, ...].unsqueeze(-1).cpu().numpy(), np.eye(4))
        nib.save(final_img, filename)
        torch.cuda.empty_cache()
        del final_img
        del converted_condition
        del original_condition
        del inverted_latents
        del synthetic_images
        
        


if __name__ == "__main__":# Load a pipeline

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(age_or_sex="age")

