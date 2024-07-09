import torch
from torch.utils.data import DataLoader
from monai.apps import DecathlonDataset
from create_dataset import *
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
import os

def compare_datasets(dataset1, dataset2, batch_size=2):
    
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)
    print("dataset1", dataset1)
    print("loader1", loader1)
    # print(loader1)
    # print("iter(loader1)", iter(loader1))
    # print("next(iter(loader1))", next(iter(loader1)))
    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    print("Dataset 1 (DecathlonDataset) output:")
    # print("batch1", batch1)
    print_batch_info(batch1)

    print("\nDataset 2 (HCPT1wDataset) output:")
    print_batch_info(batch2)

    print("\nComparison:")
    compare_batch_info(batch1, batch2)

def print_batch_info(batch):
    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, list):
                print(f"{key}: list containing file paths or other data, length {len(value)}")
    elif isinstance(batch, torch.Tensor):
        print(f"shape {batch.shape}, dtype {batch.dtype}")
    else:
        print(f"Unexpected type: {type(batch)}")

def compare_batch_info(batch1, batch2):
    if type(batch1) != type(batch2):
        print("Different batch types")
        return

    if isinstance(batch1, dict):
        keys1 = set(batch1.keys())
        keys2 = set(batch2.keys())
        if keys1 != keys2:
            print(f"Different keys: {keys1} vs {keys2}")
        for key in keys1.intersection(keys2):
            compare_tensor_info(batch1[key], batch2[key], key)
    elif isinstance(batch1, torch.Tensor):
        compare_tensor_info(batch1, batch2)
    else:
        print(f"Unexpected type: {type(batch1)}")

def compare_tensor_info(tensor1, tensor2, name=""):
    prefix = f"{name}: " if name else ""
    if tensor1.shape != tensor2.shape:
        print(f"{prefix}Different shapes: {tensor1.shape} vs {tensor2.shape}")
    if tensor1.dtype != tensor2.dtype:
        print(f"{prefix}Different dtypes: {tensor1.dtype} vs {tensor2.dtype}")



def prepare_dataloader_custom(
    args,
    batch_size,
    patch_size,
    randcrop=True,
    rank=0,
    world_size=1,
    cache=1.0,
    download=False,
    size_divisible=16,
    amp=False,
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
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
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
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            CenterSpatialCropd(keys=["image"], roi_size=val_patch_size),
            ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    os.makedirs(args.data_base_dir, exist_ok=True)
    train_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="training",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=download,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_base_dir,
        task="Task01_BrainTumour",
        section="validation",  # validation
        cache_rate=cache,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=8,
        download=download,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=val_transforms,
    )
    
    # if ddp_bool:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    # else:
    #     train_sampler = None
    #     val_sampler = None

    # train_loader = DataLoader(
    #     train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=0, pin_memory=False, sampler=train_sampler
    # )
    # val_loader = DataLoader(
    #     val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, sampler=val_sampler
    # )
    # if rank == 0:
    #     print(f'Image shape {train_ds[0]["image"].shape}')
    # return train_loader, val_loader
    return train_ds








# base_dir = '/home/sijun/meow/data_new/hcp/registered'
# hcp_dataset, _ = create_train_val_datasets(base_dir)
# decathlon_dataset = prepare_dataloader_custom()
# # train_dataset, val_dataset = create_train_val_datasets(base_dir)

# compare_datasets(decathlon_dataset, hcp_dataset)
