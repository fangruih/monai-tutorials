import pickle
import numpy as np
import os

from monai.data import DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
import torch
import random

from torch.utils.data import Dataset, DataLoader
# from create_dataset import HCPT1wDataset



def get_t1_all_file_list(file_dir_prefix = "/scr/fangruih/stru/"):
    dataset_names=["/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/metadata/abcd/paths_and_info_flexpath.pkl",
                "/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/metadata/adni_t1/paths_and_info_flexpath.pkl",
                "/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/metadata/hcp_ag_t1/paths_and_info_flexpath.pkl",
                "/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/metadata/hcp_dev_t1/paths_and_info_flexpath.pkl",
                "/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/metadata/hcp_ya_mpr1/paths_and_info_flexpath.pkl",
                "/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/metadata/ppmi_t1/paths_and_info_flexpath.pkl"]
    train_images=[]
    train_ages=[]
    val_images=[]
    val_ages=[]
    
    for dataset_name in dataset_names:
        with open(dataset_name, 'rb') as file:
            data = pickle.load(file)
            
            # Convert paths and ages to lists if they are NumPy arrays
            train_new_images = data['train']['paths'].tolist() if isinstance(data['train']['paths'], np.ndarray) else data['train']['paths']
            train_new_ages = data['train']['age'].tolist() if isinstance(data['train']['age'], np.ndarray) else data['train']['age']
            train_new_sex = data['train']['sex'].tolist() if isinstance(data['train']['sex'], np.ndarray) else data['train']['sex']
            
            val_new_images = data['val']['paths'].tolist() if isinstance(data['val']['paths'], np.ndarray) else data['val']['paths']
            val_new_ages = data['val']['age'].tolist() if isinstance(data['val']['age'], np.ndarray) else data['val']['age']
            val_new_sex = data['val']['sex'].tolist() if isinstance(data['val']['sex'], np.ndarray) else data['val']['sex']
            
            # Append new data to existing lists
            if not train_images:  # More Pythonic way to check if the list is empty
                # Direct assignment for the first file
                train_images = train_new_images
                train_ages = train_new_ages
                train_sex = train_new_sex
                
                val_images = val_new_images
                val_ages = val_new_ages
                val_sex = val_new_sex
            else:
                # Concatenation for subsequent files
                train_images += train_new_images
                train_ages += train_new_ages
                train_sex += train_new_sex
                
                val_images += val_new_images
                val_ages += val_new_ages
                val_sex += val_new_sex
            
            # Debug output to check the results
            print(train_images[-1])  # Print the last path
            
    # process z normalization for age 
    
    ages_array = np.array(train_ages)
    # Calculate mean and standard deviation
    age_mean = np.mean(ages_array)
    age_std = np.std(ages_array)
    # train_ages = (ages_array - age_mean) / age_std
    # train_ages = train_ages.tolist()
    
    # val_ages_array = np.array(val_ages)
    # age_mean = np.mean(val_ages_array)
    # val_ages = (val_ages_array - age_mean) / age_std
    # val_ages = val_ages.tolist()
    
    
    train_images = [file_dir_prefix + train_image for train_image in train_images]
    val_images = [file_dir_prefix + val_image for val_image in val_images]

    print(len(train_images))
    print(len(val_images))
    
    # Zip the conditions into one single list 
    train_conditions = [torch.tensor([a, b]) for a, b in zip(train_ages, train_sex)]
    val_conditions = [torch.tensor([a, b]) for a, b in zip(val_ages, val_sex)]
    

    
    
    
    return train_images, train_conditions, val_images, val_conditions , age_mean, age_std
    

def prepare_dataloader_from_list(
    args,
    batch_size,
    patch_size,
    
    val_ratio=0.2, 
    seed=42,
    randcrop=True,
    rank=0,
    world_size=1,
    cache=1.0,
    download=False,
    size_divisible=16,
    amp=False,
    with_conditioning= False, 
    
    cross_attention_dim=None,
    expand_token_times=None,
    
):
    ddp_bool = world_size > 1
    channel = args.channel  # 0 = Flair, 1 = T1
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
    if randcrop:
        train_crop_transform = RandSpatialCropd(keys=["image"], roi_size=patch_size, random_size=False)
        val_patch_size = [int(np.ceil(1.5 * p / size_divisible) * size_divisible) for p in patch_size]
    else:
        train_crop_transform = CenterSpatialCropd(keys=["image"], roi_size=patch_size)
        val_patch_size = patch_size

    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            # EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            train_crop_transform,
            ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            # EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            CenterSpatialCropd(keys=["image"], roi_size=val_patch_size),
            ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    os.makedirs(args.data_base_dir, exist_ok=True)
    
    
    train_images, train_conditions, val_images, val_conditions, age_mean, age_std = get_t1_all_file_list()
    
    train_ds = FileListDataset(train_images, condition_list=train_conditions, transform=train_transforms, with_conditioning=with_conditioning, 
                             cross_attention_dim=cross_attention_dim, 
                             expand_token_times= expand_token_times, compute_dtype=compute_dtype)
    
    val_ds = FileListDataset(val_images, condition_list=val_conditions,transform=val_transforms, with_conditioning=with_conditioning, 
                           cross_attention_dim=cross_attention_dim,
                           expand_token_times=expand_token_times, compute_dtype=compute_dtype)
    
    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    print("shuffle for train: ", (not ddp_bool))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=0, pin_memory=False, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, sampler=val_sampler
    )
    if rank == 0:
        # print(f'Image shape {train_ds[0]["image"]}')
        print(f'Train Image shape {train_ds[0]["image"].shape}')
        print(f'Val Image shape {val_ds[0]["image"].shape}')
    return  train_loader, val_loader #train_ds , val_ds #



class FileListDataset(Dataset):
    def __init__(self, file_list, condition_list=None,with_conditioning=False, transform=None, cross_attention_dim=None,expand_token_times=None, compute_dtype=torch.float32):
        self.file_list = file_list
        self.transform = transform
        self.with_conditioning = with_conditioning
        self.compute_dtype = compute_dtype
        self.cross_attention_dim = cross_attention_dim
        self.expand_token_times = expand_token_times
        self.condition_list= condition_list
        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        img_path = self.file_list[idx]
        data = {'image': img_path}
        
        if self.transform:
            data = self.transform(data)
            # print("data", data['image'].shape)
        
        if self.with_conditioning :
            condition_tensor = self.condition_list[idx]
            condition_tensor = condition_tensor.unsqueeze(-1)  
            condition_tensor = condition_tensor.expand(-1, self.cross_attention_dim)
            condition_tensor = condition_tensor.repeat(self.expand_token_times, 1)
            
            data['condition'] = condition_tensor
            
            
        return data
