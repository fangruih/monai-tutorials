#!/usr/bin/env python
# coding: utf-8

# Copyright (c) MONAI Consortium  
# Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
# &nbsp;&nbsp;&nbsp;&nbsp;http://www.apache.org/licenses/LICENSE-2.0  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an "AS IS" BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.
# 
# # 3D regression example based on DenseNet
# 
# This tutorial shows an example of 3D regression task based on DenseNet and array format transforms.
# 
# Here, the task is given to predict the ages of subjects from MR imagee.
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_regression/densenet_training_array.ipynb)

# ## Setup environment

# In[1]:


# !python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"


# ## Setup imports

# In[1]:


import logging
import os
import sys
import shutil
import tempfile

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    EnsureChannelFirstd,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.networks.nets import Regressor

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()


# ## Setup data directory

# In[2]:


import pickle
import numpy as np
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
        
        val_new_images = data['val']['paths'].tolist() if isinstance(data['val']['paths'], np.ndarray) else data['val']['paths']
        val_new_ages = data['val']['age'].tolist() if isinstance(data['val']['age'], np.ndarray) else data['val']['age']
        
        # Append new data to existing lists
        if not train_images:  # More Pythonic way to check if the list is empty
            # Direct assignment for the first file
            train_images = train_new_images
            train_ages = train_new_ages
            val_images = val_new_images
            val_ages = val_new_ages
        else:
            # Concatenation for subsequent files
            train_images += train_new_images
            train_ages += train_new_ages
            val_images += val_new_images
            val_ages += val_new_ages
        
        # Debug output to check the results
        print(train_images[-1])  # Print the last path
        
prefix = "/scr/fangruih/stru/"
train_images = [prefix + train_image for train_image in train_images]
val_images = [prefix + val_image for val_image in val_images]

print(len(train_images))  # Print the total number of paths loaded
print(len(train_ages))  # Print the total number of paths loaded

print(len(val_images))  # Print the total number of paths loaded
print(len(val_ages))  # Print the total number of paths loaded


# In[3]:


import numpy as np

# Path to the .npy file
file_path = "/scr/fangruih/stru/t1/hcp_ya_mpr1/169343/169343_3T_T1w_MPR1_nrm_crp.npy"
# Load the numpy file
data = np.load(file_path)

# Print the dimensions of the loaded data
print("Dimensions of the loaded data:", data.shape)


# ## Create data loaders

# In[4]:


# Define transforms
# train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(channel_dim=0), Resize((148,180,148)), RandRotate90()])
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(channel_dim=0), Resize((160, 192, 176))])
val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(channel_dim=0),Resize((160, 192, 176))])
# train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(channel_dim=0), Resize((148,180,148))])
# val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(channel_dim=0),Resize((148,180,148))])

# train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
# val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

# Define nifti dataset, data loader
check_ds = ImageDataset(image_files=train_images, labels=train_ages, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory)

im, label = monai.utils.misc.first(check_loader)
print(type(im), im.shape, label, label.shape)

# create a training data loader
train_ds = ImageDataset(image_files=train_images, labels=train_ages, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=val_images, labels=val_ages, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)


# # Focal loss (basically more weighted loss)

# In[5]:


class FocalLossRegression(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLossRegression, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the absolute error per example
        error = torch.abs(inputs - targets)
        
        # Compute the modulating factor
        modulating_factor = torch.pow(error, self.gamma)
        
        # Compute the focal loss
        loss = modulating_factor * error  # This is equivalent to error^(gamma + 1)
        
        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction


# ## Create model and train

# In[ ]:


import sys
import torch
import wandb
import numpy as np
from torch.nn import MSELoss
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
# from model import Regressor  # Assuming this is your model's import statement.# Initialize wandb
from datetime import datetime

# Get current time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

wandb.init(project="age-regressor")

# Setup the model
# model = Regressor(in_shape=[1,148,180,148], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
model = Regressor(in_shape=[1,160, 192, 176], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")

# Loss and optimizer
# loss_function = MSELoss()
gamma=2.0
loss_function = FocalLossRegression(gamma=gamma, reduction='mean')

optimizer = Adam(model.parameters(), 1e-4)

# Training settings
val_interval = 2
max_epochs = 100
best_metric = sys.float_info.max
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        if step %100==0:
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        # Log training loss every 100 stepsif step % 100 == 0:
    wandb.log({"train_loss": loss.item(), "step": epoch * len(train_loader) + step})

    # Average loss for epoch
    epoch_loss /= len(train_loader)
    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})

    # Validationif (epoch + 1) % val_interval == 0:
    model.eval()
    with torch.no_grad():
        val_losses = []
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images)
            val_loss = loss_function(val_outputs, val_labels.float())
            val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        # Check if this is the best model
        if avg_val_loss < best_metric:
            best_metric = avg_val_loss
            torch.save(model.state_dict(), f"/simurgh/u/fangruih/monai-tutorials/3d_regression/trained_model/best_metric_model_{current_time}_focal_{gamma}.pth")
            print("Saved new best model with loss:", best_metric)
            wandb.log({" best_metric":  best_metric})

print("Training completed. Best validation loss:", best_metric)
wandb.finish()


# ## Cleanup data directory
# 
# Remove directory if a temporary was used.
