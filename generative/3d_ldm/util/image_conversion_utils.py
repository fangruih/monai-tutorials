import torch
import nibabel as nib
import numpy as np
from tqdm.auto import tqdm
from generative.inferers import LatentDiffusionInferer
from torch.amp import autocast

def conversion(image, original_condition, new_condition, autoencoder, diffusion_model, diffuser_scheduler, inferer, device, save_path=None):
    """
    Perform image conversion using inversion and denoising with a new condition.
    
    Args:
        image (torch.Tensor): Input image tensor.
        original_condition (torch.Tensor): Original condition tensor (sex and age).
        new_condition (torch.Tensor): New condition tensor to apply.
        autoencoder (nn.Module): Autoencoder model.
        diffusion_model (nn.Module): Diffusion model.
        ddim_scheduler: DDIM scheduler.
        inferer: Latent diffusion inferer.
        device (torch.device): Device to run the computation on.
        save_path (str, optional): Path to save the output images. If None, images are not saved.
    
    Returns:
        tuple: (converted_image, new_condition, original_image, original_condition)
    """
    # Ensure inputs are on the correct device
    image = image.to(device)
    original_condition = original_condition.to(device)
    new_condition = new_condition.to(device)

    # Use autocast for mixed precision
    with autocast(device_type='cuda', dtype=torch.float16):
        # Encode the image to latent space
        with torch.no_grad():
            latent_clean = autoencoder.encode_stage_2_inputs(image) * 1.0

        # Inversion process
        inverted_latents = invert(
            start_latents=latent_clean,
            original_condition=original_condition,
            diffuser_scheduler=diffuser_scheduler,
            diffusion_model=diffusion_model,
            device=device
        )

        # Use the last few steps of inverted latents
        inverted_latents = inverted_latents[-10].unsqueeze(0)

        # Denoising with new condition
        with torch.no_grad():
            converted_image = inferer.sample(
                input_noise=inverted_latents,
                autoencoder_model=autoencoder,
                diffusion_model=diffusion_model,
                scheduler=ddim_scheduler,
                conditioning=new_condition
            )

    # Save images if a save path is provided
    if save_path:
        # Extract age and sex from original and new conditions
        original_age, original_sex = original_condition[0, 0, 0].item(), original_condition[0, 0, 1].item()
        new_age, new_sex = new_condition[0, 0, 0].item(), new_condition[0, 0, 1].item()
        
        # Create filenames with age and sex information
        original_filename = f"{save_path}_original_age{original_age:.1f}_sex{original_sex:.0f}.nii.gz"
        converted_filename = f"{save_path}_converted_age{new_age:.1f}_sex{new_sex:.0f}.nii.gz"
        
        # Save the images with the new filenames
        save_nifti(image[0, 0], original_filename)
        save_nifti(converted_image[0, 0], converted_filename)

    return converted_image, new_condition, image, original_condition

def invert(start_latents, original_condition, diffuser_scheduler, diffusion_model, device, num_inference_steps=50):
    latents = start_latents.clone()
    intermediate_latents = []
    diffuser_scheduler.set_timesteps(num_inference_steps)
    timesteps = reversed(diffuser_scheduler.timesteps)  

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i].unsqueeze(0)
        latent_model_input = diffuser_scheduler.scale_model_input(latents, t)
        noise_pred = diffusion_model(latent_model_input, timesteps=t, context=original_condition)

        current_t = max(0, t.item() - (1000 // num_inference_steps))
        next_t = t
        alpha_t = ddim_scheduler.alphas_cumprod[current_t].to(device)
        alpha_t_next = ddim_scheduler.alphas_cumprod[next_t]

        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

def save_nifti(tensor, filename):
    """Save a tensor as a NIfTI file."""
    nifti_image = nib.Nifti1Image(tensor.cpu().numpy(), np.eye(4))
    nib.save(nifti_image, filename)

# ... (keep any existing utility functions in the file)
