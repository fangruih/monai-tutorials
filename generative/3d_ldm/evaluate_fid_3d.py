import sys
import os
import torch
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import argparse
import logging

# Add MedicalNet directory to Python path
sys.path.append('/simurgh/u/fangruih/MedicalNet')

# Import MedicalNet modules
from setting import parse_opts
from model import generate_model

# Import your local functions
from util.dataset_utils import get_t1_all_file_list, generated_images_file_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            if file_path.endswith('.npy'):
                img = np.load(file_path)
            elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                img = nib.load(file_path).get_fdata()
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            img = torch.from_numpy(img).float()
            # print("img.shape", img.shape)
            # Ensure channel dimension is at the beginning
            if img.ndim == 3:
                img = img.unsqueeze(0)
            elif img.ndim == 4:
                if img.shape[0] != 1:
                    img = img.permute(3, 0, 1, 2)
            # Ensure the image has the correct shape (1, 160, 192, 176)
            if img.shape != (1, 160, 192, 176):
                raise ValueError(f"Unexpected image shape: {img.shape}")
            # print("img.shape after", img.shape)
            return img
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

class FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # If the model is wrapped in DataParallel, access the module
        if isinstance(original_model, torch.nn.DataParallel):
            original_model = original_model.module
        
        self.features = torch.nn.Sequential(
            *list(original_model.children())[:-1],  # all layers except conv_seg
            *list(original_model.conv_seg.children())[:-1]  # all layers of conv_seg except the last one
        )

    def forward(self, x):
        return self.features(x)

def extract_features(data_loader, model, device):
    features = []
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            if i % 100 == 0:
                logger.info(f"Processing batch {i}/{len(data_loader)}")
            batch_data = batch_data.to(device)
            # Forward pass through the model
            output = model(batch_data)
            # Flatten the output
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
    return np.concatenate(features)

def calculate_fid(real_features, generated_features):
    from pytorch_fid import fid_score
    return fid_score.calculate_fid_given_features(real_features, generated_features)
    # Ensure features are on CPU and in numpy format
    real_features = real_features.cpu().numpy() if isinstance(real_features, torch.Tensor) else real_features
    generated_features = generated_features.cpu().numpy() if isinstance(generated_features, torch.Tensor) else generated_features

    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2*covmean)
    return fid

def main():
    # Setting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # Update resume_path to use absolute path
    sets.resume_path = os.path.join('/simurgh/u/fangruih/MedicalNet', sets.resume_path)

    # Getting model
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # Move the model to the correct device
    device = torch.device(f"cuda:{sets.gpu_id[0]}" if torch.cuda.is_available() and not sets.no_cuda else "cpu")
    net = net.to(device)

    # Create feature extractor
    feature_extractor = FeatureExtractor(net)
    feature_extractor = feature_extractor.to(device)

    # Get file lists
    _, _, val_images, _, _, _ = get_t1_all_file_list()
    generated_image_paths = generated_images_file_list(sets.generated_dir)

    logger.info(f"Number of real images: {len(val_images)}")
    logger.info(f"Number of generated images: {len(generated_image_paths)}")

    # Limit the number of images to process (for both real and generated)
    max_images = 100
    val_images = val_images[:max_images]
    generated_image_paths = generated_image_paths[:max_images]

    # Create datasets and data loaders
    real_data = CustomDataset(val_images)
    generated_data = CustomDataset(generated_image_paths)

    real_loader = DataLoader(real_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    generated_loader = DataLoader(generated_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # Extract features
    logger.info("Extracting features from real images...")
    real_features = extract_features(real_loader, feature_extractor, device)
    logger.info(f"Extracted features from {len(real_features)} real images")

    logger.info("Extracting features from generated images...")
    generated_features = extract_features(generated_loader, feature_extractor, device)
    logger.info(f"Extracted features from {len(generated_features)} generated images")

    # Calculate FID
    logger.info("Calculating FID...")
    fid = calculate_fid(real_features, generated_features)
    logger.info(f"FID: {fid}")

if __name__ == '__main__':
    main()