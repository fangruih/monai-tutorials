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

def print_gpu_usage():
    print("Total memory allocated on GPU:", torch.cuda.memory_allocated(device='cuda:0') / 1e6, "MB")
    print("Total memory reserved on GPU:", torch.cuda.memory_reserved(device='cuda:0') / 1e6, "MB")

def convert_tensor_age(tensor,age):
    # Clone the tensor to avoid modifying the original
    new_tensor = tensor.clone()
    
    # Mask to identify elements that are not 0 or 1
    mask = (new_tensor != 0) & (new_tensor != 1)
    
    # Apply the mask to set the desired elements to 10
    new_tensor[mask] = age
    
    return new_tensor


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
    device="cpu",
    
):

    
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
        t = t.to(device)

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = ddim_scheduler.scale_model_input(latent_model_input, t)

        # print("Latent model input device:", latent_model_input.device)
        # print("device", device)
        # print("Timesteps device:", t.device)  # Assuming timesteps is a tensor named t
        # print("Original condition device:", original_condition.device)
        # print("Original condition device:", original_condition.device)


        noise_pred = diffusion_model(latent_model_input, timesteps=t, context=original_condition)
        

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
        
        del noise_pred
        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)