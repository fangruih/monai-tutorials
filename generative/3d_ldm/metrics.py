import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim
import math 

def calc_mses(pred, gt):
    # pred: pytorch tensor of shape batch,x,y,z or x,y,z (squeeze before!)
    # gt: pytorch tensor of shape batch,x,y,z or x,y,z (squeeze before!)
    # returns an np array of shape batch
    if len(pred.shape) == 3:
        assert len(gt.shape) == 3, f'Ground-truth images to 3D PSNR calculation had to have a shape of (x,y,z) but has {gt.shape}'
        mses = ((pred-gt)**2).mean().detach().cpu().numpy().reshape(-1)
    else:
        assert len(pred.shape) == 4, f'Predictions to 3D PSNR calculation had to have a shape of (batch,x,y,z) but has {pred.shape}'
        assert len(gt.shape) == 4, f'Ground-truth images to 3D PSNR calculation had to have a shape of (batch,x,y,z) but has {gt.shape}'
        assert pred.shape[0] == gt.shape[0], f'Predictions and ground-truth images have different batch sizes: {pred.shape[0]}, {gt.shape[0]}'
        mses = ((pred-gt)**2).mean((1,2,3)).detach().cpu().numpy() # an np array of shape batch
    return mses

def calc_3d_psnrs_from_mses(mses):
    # mses: numpy array of shape batch (calculated from pred & gt)
    # NOTE: assumes that gt had the range [0,1]! Thus, 20log(1)=0!
    return -10 * np.log10(mses)

def calc_ssims(pred, gt):
    # pred: pytorch tensor of shape batch, ... (no need to squeeze)
    # gt: pytorch tensor of shape batch, ... (no need to squeeze)
    # returns an np array of shape batch
    # NOTE: assumes that gt has the range [0,1]!
    assert pred.shape[0] == gt.shape[0], 'Batch sizes of prediction and gt are different in ssim calculation!'
    ssim_val = ssim(pred, gt, data_range=1, size_average=False) # return (N,)
    return ssim_val.detach().cpu().numpy()

def calc_ms_ssims(pred, gt):
    # pred: pytorch tensor of shape batch, ... (no need to squeeze)
    # gt: pytorch tensor of shape batch, ... (no need to squeeze)
    # returns an np array of shape batch
    # NOTE: assumes that gt has the range [0,1]!
    assert pred.shape[0] == gt.shape[0], 'Batch sizes of prediction and gt are different in ms-ssim calculation!'
    ms_ssim_val = ms_ssim(pred, gt, data_range=1, size_average=False ,win_size=7) #(N,)
    return ms_ssim_val.detach().cpu().numpy()


def calc_mmd(pred, gt):
    # pred: pytorch tensor of shape batch, ... (no need to squeeze)
    # gt: pytorch tensor of shape batch, ... (no need to squeeze)
    # returns a number
    # if you will take an average, weigh the dist with the batch size in case of different batch sizes
    # NOTE: MMD estimation requires great batch sizes to approximate it better! Save predictions and calculate MMD at the end.
    assert pred.shape[0] == gt.shape[0], 'Batch sizes of prediction and gt are different in mmd calculation!'
    batch = pred.shape[0]
    x = pred.view(batch, -1)
    y = gt.view(batch, -1)

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    xx = xx - torch.diag(torch.diag(xx))
    yy = yy - torch.diag(torch.diag(yy))
    # print("batch", batch)
    # print("(batch*(batch-1))", (batch*(batch-1)))
    beta = (1./(batch*(batch-1)))
    gamma = (2./(batch*batch)) 

    dist = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)
    print("dist", dist)
    return math.sqrt(dist.item()) # number

def metrics_mean_mses_psnrs_ssims_mmd(pred,gt):
    mses= calc_mses(pred.squeeze(), gt.squeeze())
    # print("mses", mses)
    metric_3d_psnrs_from_mses= calc_3d_psnrs_from_mses(mses)
    # print("psnrs", metric_3d_psnrs_from_mses)
    ssims= calc_ms_ssims(pred, gt)
    # print("ssims", ssims)
    #mmd= calc_mmd(pred, gt)
    # print("mmd", mmd)

    # Calculating the mean for each metric
    mean_mses = np.mean(mses)
    mean_psnrs = np.mean(metric_3d_psnrs_from_mses)
    mean_ssims = np.mean(ssims)
    # MMD returns a single number already, no need to mean
    
    return mean_mses, mean_psnrs, mean_ssims #, mmd