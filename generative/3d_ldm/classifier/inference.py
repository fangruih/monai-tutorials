import argparse
import os
import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import Image_Sex_Dataset, MultiEpochsDataLoader#, AverageMeter
# from discriminator import weights_init_normal, SFCNDiscriminator
# from generator import DiffeoGenerator
# from losses import r1_reg, smooth_loss_l2
from dp_model.model_files.sfcn import SFCN

import sys
import time
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import nibabel as nib

random.seed(1337)

def inference(imgs, model_path):
    if torch.cuda.is_available():
        print('Cuda is available')
        device = torch.device('cuda')
    else:
        print('Cuda is not available')
        device = torch.device('cpu')
        
    model = SFCN(output_dim=2, channel_number=[28, 58, 128, 256, 256, 64]).to(device)
    
    #load the model weights from model_path
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set the model to evaluation mode
    model.eval()
    
    imgs = imgs.to(device).float()  # Make sure imgs is a numpy array when using torch.from_numpy()
    
    
    with torch.no_grad():  # Do not track gradients
        prediction = model(imgs)
    print(prediction)
    



if __name__ == '__main__':
    nifti_file_path_convert = '/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/output/abcd/converted_20240807_093837.nii'
    imgs = nib.load(nifti_file_path_convert)
    imgs = imgs.get_fdata()
    imgs = torch.from_numpy(imgs)
    imgs = imgs.squeeze()
    imgs = imgs.unsqueeze(0).unsqueeze(0).float()
    print("imgs.shape", imgs.shape)
    model_path= "/simurgh/u/fangruih/monai-tutorials/generative/3d_ldm/classifier/sex_classification_model.pth"
    inference(imgs=imgs, model_path=model_path)
    
    
    
    