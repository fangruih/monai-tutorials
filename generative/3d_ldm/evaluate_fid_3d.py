import sys
import os
import torch
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import argparse


# Add MedicalNet directory to Python path
sys.path.append('/simurgh/u/fangruih/MedicalNet')

# Import MedicalNet modules
from setting import parse_opts 
# from setting import parse_opts as parse_opts_medicalnet

from model import generate_model

# Import your local functions
from util.dataset_utils import get_t1_all_file_list, generated_images_file_list


# def parse_opts():
#     medicalnet_args = parse_opts_medicalnet()
    
#     # Manually add the generated_dir argument
#     parser = argparse.ArgumentParser(description='FID Evaluation')
#     parser.add_argument('--generated_dir', default='', type=str, help='Directory containing generated images')
    
#     # Parse only the generated_dir argument
#     generated_dir_arg, _ = parser.parse_known_args()
    
#     # Add the generated_dir to the medicalnet_args
#     medicalnet_args.generated_dir = generated_dir_arg.generated_dir
    
#     return medicalnet_args
class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = nib.load(self.file_paths[idx]).get_fdata()
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)  # Add channel dimension
        return img

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

def extract_features(data_loader, model, sets):
    features = []
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            if not sets.no_cuda:
                batch_data = batch_data.cuda()
            # Forward pass through the model
            output = model(batch_data)
            # Flatten the output
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
    return np.concatenate(features)

def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
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

    # Create feature extractor
    feature_extractor = FeatureExtractor(net)
    if not sets.no_cuda:
        feature_extractor = feature_extractor.cuda()

    # Get file lists
    _, _, val_images, _, _, _ = get_t1_all_file_list()
    generated_image_paths = generated_images_file_list(sets.generated_dir)

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
    real_features = extract_features(real_loader, feature_extractor, sets)
    generated_features = extract_features(generated_loader, feature_extractor, sets)

    # Calculate FID
    fid = calculate_fid(real_features, generated_features)
    print(f"FID: {fid}")

if __name__ == '__main__':
    main()