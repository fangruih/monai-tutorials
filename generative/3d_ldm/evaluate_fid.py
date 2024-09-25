import os
import numpy as np
import nibabel as nib
from monai.metrics import FIDMetric
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from scipy import linalg
import numpy as np
# def load_and_preprocess_images(file_paths, target_sizes=None):
#     images = {
#         'dim1': [],
#         'dim2': [],
#         'dim3': []
#     }
#     for path in tqdm(file_paths, desc="Loading and preprocessing images"):
#         if path.endswith('.npy'):
#             data = np.load(path)
#         elif path.endswith('.nii'):
#             img = nib.load(path)
#             data = img.get_fdata()
#         else:
#             raise ValueError(f"Unsupported file format: {path}")
        
#         # Extract middle slices from each dimension
#         slices = [
#             data[data.shape[0]//2, :, :],
#             data[:, data.shape[1]//2, :],
#             data[:, :, data.shape[2]//2]
#         ]
        
#         # Normalize and resize (if needed) each slice
#         for i, (slice, target_size) in enumerate(zip(slices, target_sizes or [None]*3)):
#             # Normalize
#             normalized_slice = (slice - slice.min()) / (slice.max() - slice.min())
#             print(f"normalized_slice.shape: {normalized_slice.shape}")
#             print(f"target_size: {target_size}")
#             # Resize if target_size is provided
#             if target_size and normalized_slice.shape != target_size:
#                 diff_y = normalized_slice.shape[0] - target_size[0]
#                 diff_x = normalized_slice.shape[1] - target_size[1]
#                 print(f"normalized_slice.shape: {normalized_slice.shape}, target_size: {target_size}")
#                 print(f"diff_y: {diff_y}, diff_x: {diff_x}")
                
#                 start_y = diff_y // 2 if diff_y > 0 else 0
#                 start_x = diff_x // 2 if diff_x > 0 else 0
                
#                 end_y = start_y + target_size[0]
#                 end_x = start_x + target_size[1]
                
#                 # Crop or pad
#                 if diff_y > 0 or diff_x > 0:
#                     resized_slice = normalized_slice[start_y:end_y, start_x:end_x]
#                 else:
#                     resized_slice = np.pad(normalized_slice, 
#                                            ((max(0, -start_y), max(0, target_size[0]-normalized_slice.shape[0]-start_y)),
#                                             (max(0, -start_x), max(0, target_size[1]-normalized_slice.shape[1]-start_x))),
#                                            mode='constant')
#             else:
#                 resized_slice = normalized_slice
            
#             images[f'dim{i+1}'].append(resized_slice)
        
#     # Convert to numpy arrays
#     for dim in ['dim1', 'dim2', 'dim3']:
#         images[dim] = np.array(images[dim])
#         print(f"Size of {dim} slices: {images[dim].shape}")
    
#     return images['dim1'], images['dim2'], images['dim3']
def load_and_preprocess_images(file_paths):
    images = {
        'dim1': [],
        'dim2': [],
        'dim3': []
    }
    for path in tqdm(file_paths, desc="Loading and preprocessing images"):
        
        if path.endswith('.npy'):
            data = np.load(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            img = nib.load(path)
            data = img.get_fdata()
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        # print("data.shape: ", data.shape)
        # Extract middle slices from each dimension
        # Squeeze data to remove any singleton dimensions
        
        data = np.squeeze(data)
        slices = [
            data[data.shape[0]//2, :, :],
            data[:, data.shape[1]//2, :],
            data[:, :, data.shape[2]//2]
        ]
        
        # Normalize each slice
        for i, slice in enumerate(slices):
            # Normalize
            normalized_slice = (slice - slice.min()) / (slice.max() - slice.min())
            images[f'dim{i+1}'].append(normalized_slice)
        
    # Convert to numpy arrays
    for dim in ['dim1', 'dim2', 'dim3']:
        images[dim] = np.array(images[dim])
        print(f"Size of {dim} slices: {images[dim].shape}")
    
    return images['dim1'], images['dim2'], images['dim3']
# def calculate_fid(real_images, generated_images):
#     fid_metric = FIDMetric()
#     real_tensor = torch.from_numpy(real_images).float()
#     generated_tensor = torch.from_numpy(generated_images).float()
#     fid_score = fid_metric(real_tensor, generated_tensor)
#     return fid_score.item()
# def calculate_fid(real_images, generated_images):
#     # Ensure images are in the correct format (B, C, H, W)
#     print("Entering calculate_fid function")
#     print("real_images shape:", real_images.shape)
#     print("generated_images shape:", generated_images.shape)

#     real_images = real_images.squeeze()
#     generated_images = generated_images.squeeze()
#     # print("real_images.shape: ", real_images.shape)
#     # print("generated_images.shape: ", generated_images.shape)
#     if real_images.ndim == 3:
#         real_images = real_images[:, np.newaxis, :, :]
#     if generated_images.ndim == 3:
#         generated_images = generated_images[:, np.newaxis, :, :]
#     print("After reshaping:")
#     print("real_images shape:", real_images.shape)
#     print("generated_images shape:", generated_images.shape)

#     # Convert to PyTorch tensors and move to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     real_tensor = torch.from_numpy(real_images).float().to(device)
#     generated_tensor = torch.from_numpy(generated_images).float().to(device)
def calculate_fid(real_images, generated_images):
    # Ensure images are in the correct format (B, C, H, W)
    print("Entering calculate_fid function")
    print("real_images shape:", real_images.shape)
    print("generated_images shape:", generated_images.shape)

    real_images = real_images.squeeze()
    generated_images = generated_images.squeeze()
    
    if real_images.ndim == 3:
        real_images = real_images[:, np.newaxis, :, :]
    if generated_images.ndim == 3:
        generated_images = generated_images[:, np.newaxis, :, :]
    print("After reshaping:")
    print("real_images shape:", real_images.shape)
    print("generated_images shape:", generated_images.shape)
    


    # Convert to PyTorch tensors and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_tensor = torch.from_numpy(real_images).float().to(device)
    generated_tensor = torch.from_numpy(generated_images).float().to(device)
    print("real_tensor shape before :", real_tensor.shape)
    if real_tensor.shape[1] == 1:
        real_tensor = real_tensor.repeat(1, 3, 1, 1)
    if generated_tensor.shape[1] == 1:
        generated_tensor = generated_tensor.repeat(1, 3, 1, 1)
    
    print("real_tensor shape after:", real_tensor.shape)
    # Resize images to 299x299 if they're not already that size
    if real_tensor.shape[2:] != (299, 299):
        real_tensor = nn.functional.interpolate(real_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    if generated_tensor.shape[2:] != (299, 299):
        generated_tensor = nn.functional.interpolate(generated_tensor, size=(299, 299), mode='bilinear', align_corners=False)

    # Load pre-trained Inception v3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Extract features
    def get_features(images):
        with torch.no_grad():
            features = inception_model(images)
        return features.cpu().numpy()

    real_features = get_features(real_tensor)
    generated_features = get_features(generated_tensor)

    # Calculate mean and covariance
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return fid
    

def evaluate_fid(real_image_paths, generated_image_paths):
    print("Loading and preprocessing generated images...")
    # generated_dim = load_and_preprocess_images(generated_image_paths)
    generated_dim1, generated_dim2, generated_dim3 = load_and_preprocess_images(generated_image_paths)
    # print("generated_dim1.shape: ", generated_dim1.shape)
    # print("generated_dim2.shape: ", generated_dim2.shape)
    # print("generated_dim3.shape: ", generated_dim3.shape)
    # Get the sizes of generated images for each dimension to use as target sizes for real images
    target_size_dim1 = generated_dim1.shape[1:]
    target_size_dim2 = generated_dim2.shape[1:]
    target_size_dim3 = generated_dim3.shape[1:]
    target_sizes = [target_size_dim1, target_size_dim2, target_size_dim3]
    
    print("Loading and preprocessing real images...")
    print(f"target_sizes: {target_sizes}")
    
    real_dim1, real_dim2, real_dim3 = load_and_preprocess_images(real_image_paths)#, target_sizes)
    # real_dim = load_and_preprocess_images(real_image_paths)#, target_sizes)
    
    print("real_dim1.shape: ", real_dim1.shape)
    print("real_dim2.shape: ", real_dim2.shape)
    print("real_dim3.shape: ", real_dim3.shape)
    print("Calculating FID score...")
    fid_scores = []
    for i in range(3):  # Calculate FID for each middle slice
        print(f"real_dim{i+1}.shape: {real_dim1.shape}")
        fid_score = calculate_fid(eval(f"real_dim{i+1}"), eval(f"generated_dim{i+1}"))
        # fid_score = calculate_fid(real_dim[i],generated_dim[i])
        
        fid_scores.append(fid_score)
        print(f"FID Score for dimension {i+1}: {fid_score}")
    
    print("FID Scores for each dimension:")
    for i, score in enumerate(fid_scores):
        print(f"FID Score Dimension {i+1}: {score}")
    avg_fid_score = np.mean(fid_scores)
    print(f"Average FID Score: {avg_fid_score}")
    return {
        "dim1_fid": fid_scores[0],
        "dim2_fid": fid_scores[1],
        "dim3_fid": fid_scores[2],
        "avg_fid": avg_fid_score
    }

# Example usage:
# real_image_paths = ['/path/to/real/image1.nii.gz', '/path/to/real/image2.nii.gz', ...]
# generated_image_paths = ['/path/to/generated/image1.nii.gz', '/path/to/generated/image2.nii.gz', ...]
# avg_fid_score = evaluate_fid(real_image_paths, generated_image_paths)
from util.dataset_utils import get_t1_all_file_list, generated_images_file_list

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate FID score between real and generated images")
    # parser.add_argument("--real", type=str, required=True, help="Path to directory containing real images")
    parser.add_argument("--generated", type=str, required=True, help="Path to directory containing generated images")
    args = parser.parse_args()

    train_images, train_conditions, val_images, val_conditions, age_mean, age_std = get_t1_all_file_list()
    
    # print("args.generated: ", args.generated)
    # generated_image_paths = generated_images_file_list(args.generated)
    # print("generated_image_paths: ", generated_image_paths)
    real_image_paths = val_images[:100]
    
    # Check if generated argument is provided, if not use default path
    if args.generated is None:
        args.generated = "/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/output/t1_all/counterfactual/age/20240924_223421"
        print(f"No generated directory provided. Using default path: {args.generated}")

    # Ensure the generated directory exists
    if not os.path.exists(args.generated):
        print(f"Error: The specified generated directory does not exist: {args.generated}")
        return

    # Get the list of generated image paths
    generated_image_paths = generated_images_file_list(args.generated)
    
    generated_image_paths = generated_image_paths[:100]
    # Limit the number of images to process (for both real and generated)
    max_images = 100
    real_image_paths = real_image_paths[:max_images]
    generated_image_paths = generated_image_paths[:max_images]

    print(f"Using {len(real_image_paths)} real images and {len(generated_image_paths)} generated images for FID calculation.")
    if len(real_image_paths) == 0 or len(generated_image_paths) == 0:
        print("Error: No .nii.gz files found in one or both directories.")
        return

    print(f"Found {len(real_image_paths)} real images and {len(generated_image_paths)} generated images.")
    
    avg_fid_score = evaluate_fid(real_image_paths, generated_image_paths)
    print(f"Final Average FID Score: {avg_fid_score}")

if __name__ == "__main__":
    main()



