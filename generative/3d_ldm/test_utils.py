import torch
from torch.utils.data import DataLoader
from monai.apps import DecathlonDataset
from create_dataset import *

def compare_datasets(dataset1, dataset2, batch_size=1):
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    print("Dataset 1 (DecathlonDataset) output:")
    print_batch_info(batch1)

    print("\nDataset 2 (HCPT1wDataset) output:")
    print_batch_info(batch2)

    print("\nComparison:")
    compare_batch_info(batch1, batch2)

def print_batch_info(batch):
    if isinstance(batch, dict):
        for key, value in batch.items():
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
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

# Usage
decathlon_dataset = DecathlonDataset(
    root_dir="/home/sijun/meow/monai-tutorials/generative/3d_ldm/dataset",
    task="Task01_BrainTumour",
    section="training",
    cache_rate=1.0,
    num_workers=8,
    download=False,
    seed=0,
    # transform=train_transforms,
)




base_dir = '/home/sijun/meow/data/hcp_new/hcp/registered'
hcp_dataset, _ = create_train_val_datasets(base_dir)
# train_dataset, val_dataset = create_train_val_datasets(base_dir)

compare_datasets(decathlon_dataset, hcp_dataset)
