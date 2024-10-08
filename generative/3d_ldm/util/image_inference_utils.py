import torch
import nibabel as nib
import numpy as np
from generative.inferers import LatentDiffusionInferer
from torch.cuda.amp import autocast

def generate_image_from_condition(condition, autoencoder, diffusion_model, scheduler, inferer, device, noise_shape, save_path=None):
    """
    Generate an image from a given condition using the latent diffusion model.

    Args:
        condition (torch.Tensor): Condition tensor (sex and age).
        autoencoder (nn.Module): Autoencoder model.
        diffusion_model (nn.Module): Diffusion model.
        scheduler: Noise scheduler.
        inferer: Latent diffusion inferer.
        device (torch.device): Device to run the computation on.
        latent_shape (list): Shape of the latent space.
        save_path (str, optional): Path to save the output image. If None, image is not saved.

    Returns:
        torch.Tensor: Generated image.
    """
    # Ensure condition is on the correct device
    condition = condition.to(device)

    # Generate random noise
    # noise_shape = [1, autoencoder.latent_channels] + latent_shape
    noise = torch.randn(noise_shape).to(device)

    # Generate image
    with torch.no_grad():
        generated_image = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            conditioning=condition
        )

    # Save image if a save path is provided
    if save_path:
        save_nifti(generated_image[0, 0], save_path)

    return generated_image

def save_nifti(tensor, filename):
    """Save a tensor as a NIfTI file."""
    nifti_image = nib.Nifti1Image(tensor.cpu().numpy(), np.eye(4))
    nib.save(nifti_image, filename)

# ... (keep any existing utility functions in the file)
