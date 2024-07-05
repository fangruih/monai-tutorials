import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path

class HCPT1wDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = nib.load(img_path).get_fdata()
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)

        return image




import random
from torch.utils.data import DataLoader

def create_train_val_datasets(base_dir, val_ratio=0.2, seed=42):
    base_path = Path(base_dir)
    all_files = list(base_path.rglob('*/**/3T/T1w_MPR1/*_3T_T1w_MPR1.nii.gz'))
    
    # Shuffle the file list
    random.seed(seed)
    random.shuffle(all_files)
    
    # Split into train and validation
    val_size = int(len(all_files) * val_ratio)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]
    
    # Create datasets
    train_dataset = HCPT1wDataset(train_files)
    val_dataset = HCPT1wDataset(val_files)
    
    return train_dataset, val_dataset

# Usage
base_dir = '/home/sijun/meow/data/hcp_new/hcp/registered'
train_dataset, val_dataset = create_train_val_datasets(base_dir)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
