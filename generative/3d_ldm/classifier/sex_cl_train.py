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

# from monai.transforms import *
# from monai import transforms as transforms
# from monai.data.dataloader import DataLoader as MonaiDataLoader

random.seed(1337)

# Tensor type
if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor


'''
output = model(input_data)

# Output, loss, visualisation
x = output[0].cpu().reshape([1, -1])
loss = F.nll_loss(x, y)

# Prediction, Visualisation and Summary
x = np.exp(x.numpy().reshape(-1))
'''
def train(opt):
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H:%M:%S")
    print("date and time =", dt_string)
    dir_name = f"saved_models/sex_cls_net_{dt_string}"
    os.makedirs(dir_name, exist_ok=True)
    log_file = open(f'{dir_name}/log.txt', 'a')
    log_file.write('BEGINNING TRAINING...')
    log_file.write(f'Opt:{opt}')
    log_file.close()
    log_file = open(f'{dir_name}/log.txt', 'a')
    if torch.cuda.is_available():
        print('Cuda is available')
        device = torch.device(f"cuda:{opt.gpu_id}")
    else:
        print('Cuda is not available')
        device = torch.device('cpu')

    # Loss functions
    criterion_cls = F.binary_cross_entropy_with_logits
    # Initialization
    model = SFCN(output_dim=2, channel_number=[28, 58, 128, 256, 256, 64]).to(device)


    # Multi-GPU training
    if opt.n_gpu > 1:
        model = nn.DataParallel(model, list(range(opt.n_gpu)))
    model.apply(weights_init_normal)
    
    

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    tr_dataset = Image_Sex_Dataset('train', aug_p=opt.aug_p)
    tr_dataloader = MultiEpochsDataLoader(tr_dataset,
                                       batch_size=opt.batch_size,
                                       drop_last=False,
                                       shuffle=True,
                                       num_workers=opt.n_cpu,
                                       pin_memory=True)
    vl_dataset = Image_Sex_Dataset('val', aug_p=opt.aug_p)
    vl_dataloader = MultiEpochsDataLoader(vl_dataset,
                                       batch_size=opt.batch_size,
                                       drop_last=False,
                                       shuffle=True,
                                       num_workers=opt.n_cpu,
                                       pin_memory=True)
    # test_transforms = transforms.Compose([transforms.NormalizeIntensity()])
    # monai_transforms = [transforms.Rand3DElastic(sigma_range=(0.01, 1), magnitude_range=(0, 1),
    #                                         prob=opt.aug_p, rotate_range=(0.18, 0.18, 0.18),
    #                                         translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
    #                                         spatial_size=None, padding_mode="border", as_tensor_output=False),
    #                         transforms.RandHistogramShift(num_control_points=(5, 15), prob=opt.aug_p),
    #                         transforms.RandAdjustContrast(prob=opt.aug_p),
    #                         transforms.RandGaussianNoise(prob=opt.aug_p),
    #                         transforms.NormalizeIntensity()]
    # monai_transforms = transforms.Compose(monai_transforms)
    
    best_val_acc = 0
    st_time = time.time()
    for epoch in range(opt.n_epochs):
        str_ep_beg = f'Ep {epoch} began, t elapsed in the pr. ep.: {time.time()-st_time}'
        print(str_ep_beg)
        log_file.write(str_ep_beg)
        st_time = time.time()
        tr_losses = []
        model.train()
        print('len of train set', len(tr_dataloader.dataset))
        for i, (imgs, sexes) in enumerate(tr_dataloader):
            if i % 100 == 0:
                print(f'i equals:{i}, time elapsed:{time.time()-st_time}')
                torch.save(model.state_dict(), f"{dir_name}/ep{epoch}_{i}.pth")

            st_time = time.time()
            
            # Model inputs
            imgs = Variable(imgs.type(Tensor)).to(device)
            sexes = Variable(sexes.type(Tensor)).to(device).long()
            #print('imgs loaded, shape', imgs.shape)
        #     if opt.aug_p > 0:
        #         imgs = monai_transforms(imgs)
        #    # print(f'transformed image shape {image.shape}')
        #     else:
        #         imgs = test_transforms(imgs)
            #print(f'test transformed image shape {image.shape}')
            #print(f'imgs shape:{imgs.shape}, sexes.shape:{sexes.shape}, time elapsed in the pr. ep.:{time.time()-st_time}')
            #st_time = time.time()

            optimizer.zero_grad()

            #imgs.requires_grad_()
            outs = model(imgs).squeeze() #x_real: (128,1,64,64,64) | out_real: () 

            #print(f'outs.shape:{outs.shape}, time elapsed:{time.time()-st_time}')
            #st_time = time.time()
            #print('outs bu ', outs)
            #print('sexes bu ', sexes)
            loss = F.nll_loss(outs, sexes)
            #print(f'loss.shape:{loss.shape}, time elapsed:{time.time()-st_time}')
            #st_time = time.time()

            tr_losses.append(loss.detach().cpu().numpy().item())
            #print(f"i {i} lss {tr_losses[-1]}")
            loss.backward()
            #print(f'after loss backward, time elapsed:{time.time()-st_time}')
            #st_time = time.time()
            # print('params:')
            # for param in model.parameters():
            #     print(param.grad)
            # print('done')
            optimizer.step()
            

            #print(f'after step, time elapsed:{time.time()-st_time}')
            #st_time = time.time()

            #progress_bar.set_postfix(loss=tr_losses[-1])
            #progress_bar.update(x_real.size(0))
        # check val performance
        print('ep tr ended, val starting')
        model.eval()
        val_acc = 0
        lft_preds = 0
        print('len of val set', len(vl_dataloader.dataset))
        for i, (imgs, sexes) in enumerate(vl_dataloader):
            sexes = sexes.cpu().numpy()
            #print(f'type {sexes.dtype} sh {sexes.shape}')
            # Model inputs
            imgs = Variable(imgs.type(Tensor)).to(device) # [5, 1, 197, 233, 189]
            #sexes = Variable(sexes.type(Tensor)).long()

                # Don't forget this. BatchNorm will be affected if not in eval mode.
            with torch.no_grad():
                output = model(imgs).squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().numpy()
          #  print('md out shape', model(imgs).shape)
           # print('out dhaspe', output.shape)
            #print(output)
            # Prediction, Visualisation and Summary
            preds = np.argmax(output, 1)
            #print(preds)
            #print('preds sghape', preds.shape)
            #print('sum', (sexes==preds).sum())
            #print((sexes==preds))
            val_acc += (sexes==preds).sum()
            lft_preds += (preds==0).sum()
        val_acc_str = f'Val true preds: {val_acc} / {len(vl_dataloader.dataset)} = {val_acc / len(vl_dataloader.dataset)}, Val left preds: {lft_preds}'
        print(val_acc_str)
        log_file.write(val_acc_str)
        log_file.close()
        log_file = open(f'{dir_name}/log.txt', 'a')
        val_acc = val_acc / len(vl_dataloader.dataset)
        if val_acc > best_val_acc:
            print(f'BEST VAL ACC is {val_acc}.')
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{dir_name}/ep{epoch}_vacc{val_acc}.pth")
    print('Training ended')
    log_file.close()

def weights_init_normal(model):
    '''
    More of a stable init for this problem than the default Pytorch init
    :param model: the model to initialise
    :return:
    '''
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)


if __name__ == '__main__':
    print('Main method started!')
    print('Python version', sys.version)
    print('Cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Device name:', torch.cuda.get_device_name(0))
        t = torch.cuda.get_device_properties(0).total_memory
        print('Total device memory:', t)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-2, help="adam: learning rate")

    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--aug_p", type=float, default=0.8, help="amount of augmentation to use")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus on machine to use")
    parser.add_argument("--img_dims", type=int, default=[197, 233, 189], help="image dimensions")
    parser.add_argument("--gpu_id", type=int, default=1, help="image dimensions")
    opt = parser.parse_args()
    #opt.gpu_id = 0
    train(opt)
