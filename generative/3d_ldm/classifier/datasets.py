import numpy as np

from torch.utils.data import Dataset
from monai.transforms import *
from monai import transforms as t
from monai.data.dataloader import DataLoader as MonaiDataLoader

from torch.autograd import Variable

import glob
import pandas as pd
import pickle
import torch
from dp_model import dp_utils as dpu
#import nibabel as nib

class Image_Handedness_Dataset(Dataset):
    '''
    This is an example Dataset class that you should adapt to use your own data
    here we produce just random tensors, to illustrate the augmentation pipeline and how it would fit in
    the train_model.py code. For the paper we used UK Biobank data which is freely avaible, but not shareable
    '''

    def __init__(self, subset, aug_p=0.8, gpu_id = None, desired_handedness = None):
        '''
        Recommended to add oversampling to ensure the generator learns the task properly
        :param aug_p: The augmentation probability. Due to the difficulty of the task and the lack of brain imaging
        available it is recommended that this be set to a fairly high value.
        '''

        self.aug_p = aug_p
        self.subset = subset
        self.test_transforms = t.Compose([t.NormalizeIntensity()])
        self.monai_transforms = [ToTensor(),
                                 t.Rand3DElastic(sigma_range=(0.01, 1), magnitude_range=(0, 1),
                                                 prob=aug_p, rotate_range=(0.18, 0.18, 0.18),
                                                 translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
                                                 spatial_size=None, padding_mode="border", as_tensor_output=False),
                                 t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
                                 t.RandAdjustContrast(prob=aug_p),
                                 t.RandGaussianNoise(prob=aug_p),
                                 t.NormalizeIntensity()]
        self.monai_transforms = t.Compose(self.monai_transforms)
        self.gpu_id = gpu_id
        self.desired_handedness = desired_handedness
        
        #dict_paths = [dict_paths[-1]]
        # for dict_path in dict_paths:
        #     with open(dict_path, 'rb') as f:
        #         curr_dict = pickle.load(f)
        #         print(f'dataset {dict_path}')
        #         spp = np.load(curr_dict['train']['paths'][0]).shape
        #         print(f'shape {spp} b')
        #     print('sdsd')
        self.data = {}
        keys = ['paths', 'handedness', 'handedness_value'] #'ages' 'handednesses'
        for dict_path in dict_paths:
            with open(dict_path, 'rb') as f:
                curr_dict = pickle.load(f)
                for key in keys:
                    if key in curr_dict[subset]:
                        if key in self.data:
                            self.data[key] = np.concatenate((self.data[key], curr_dict[subset][key]))
                        else:
                            self.data[key] = curr_dict[subset][key]
                    else:
                        emp_arr=np.empty((curr_dict[subset]['paths'].shape[0]))
                        emp_arr[:] = None
                        if key in self.data:
                            self.data[key] = np.concatenate((self.data[key], emp_arr))
                        else:
                            self.data[key] = emp_arr
        if self.desired_handedness is not None:
            new_data = {}
            for i in range(len(self.data['paths'])):
                if self.data['handedness'][i] == desired_handedness:
                    if 'paths' in new_data:
                        new_data['paths'].append(self.data['paths'][i])
                    else:
                        new_data['paths'] = [self.data['paths'][i]]
                    if 'handedness' in new_data:
                        new_data['handedness'].append(self.data['handedness'][i])
                    else:
                        new_data['handedness'] = [self.data['handedness'][i]]
                    if 'handedness_value' in new_data:
                        new_data['handedness_value'].append(self.data['handedness_value'][i])
                    else:
                        new_data['handedness_value'] = [self.data['handedness_value'][i]]
            new_data['paths'] = np.array(new_data['paths'])
            new_data['handedness'] = np.array(new_data['handedness'])
            new_data['handedness_value'] = np.array(new_data['handedness_value'])
            self.data = new_data




    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, index):
        #print(f'dataset getitem subset {self.subset}')
        #image = np.expand_dims(self.brain_volumes[index], axis=0)
        
        image = np.load(self.data['paths'][index])#).to(f"cuda:{self.gpu_id}")
        #image = torch.from_numpy(image).type(torch.cuda.FloatTensor)#.to(f"cuda:{self.gpu_id}")
        #print(f'image shape {image.shape}')
        # if self.subset == 'train' and self.aug_p > 0:
        #     image = self.monai_transforms(image)
        #    # print(f'transformed image shape {image.shape}')
        # else:
        #     image = self.test_transforms(image)
            #print(f'test transformed image shape {image.shape}')
        if self.desired_handedness is None:
            return image, self.data['handedness'][index], self.data['handedness_value'][index] #, self.data['paths'][index] #self.data['ages'][index], self.data['handedness'][index], index
        else:
            return image, self.data['handedness_value'][index] #, self.data['paths'][index] #self.data['ages'][index], self.data['handedness'][index], index

class Image_Age_Dataset(Dataset):
    '''
    This is an example Dataset class that you should adapt to use your own data
    here we produce just random tensors, to illustrate the augmentation pipeline and how it would fit in
    the train_model.py code. For the paper we used UK Biobank data which is freely avaible, but not shareable
    '''

    def __init__(self, subset, aug_p=0.8):
        '''
        Recommended to add oversampling to ensure the generator learns the task properly
        :param aug_p: The augmentation probability. Due to the difficulty of the task and the lack of brain imaging
        available it is recommended that this be set to a fairly high value.
        '''

        self.aug_p = aug_p
        self.subset = subset
        self.test_transforms = t.Compose([t.NormalizeIntensity()])
        self.monai_transforms = [ToTensor(),
                                 t.Rand3DElastic(sigma_range=(0.01, 1), magnitude_range=(0, 1),
                                                 prob=aug_p, rotate_range=(0.18, 0.18, 0.18),
                                                 translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
                                                 spatial_size=None, padding_mode="border", as_tensor_output=False),
                                 t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
                                 t.RandAdjustContrast(prob=aug_p),
                                 t.RandGaussianNoise(prob=aug_p),
                                 t.NormalizeIntensity()]
        self.monai_transforms = t.Compose(self.monai_transforms)

        # load HCP T1
        dict_paths = ['/home/ridvan/data/data_paths/hcp/registered/HCP_T1w_1_2_with_age_sex_handedness.pkl', \
                      '/home/ridvan/data/data_paths/hcp/registered/HCP_T2w_1_2_with_age_sex_handedness.pkl', \
                      '/home/ridvan/data/data_paths/adni/registered/ADNI_T1w_with_age_sex.pkl',\
                      '/home/ridvan/data/data_paths/ppmi/registered/PPMI_T1-anatomical_age_sex_group.pkl',\
                      '/home/ridvan/data/data_paths/ppmi/registered/PPMI_T2_in_T1-anatomical_space_age_sex_group.pkl',\
                       '/home/ridvan/data/data_paths/openneuro/registered/openneuro_acq-CUBE_T2w_age_sex_handedness.pkl',\
                        '/home/ridvan/data/data_paths/openneuro/registered/openneuro_acq-MPRAGE_T1w_age_sex_handedness.pkl']
        
        # for dict_path in dict_paths:
        #     with open(dict_path, 'rb') as f:
        #         curr_dict = pickle.load(f)
        #         print(f'dataset {dict_path}')
        #         spp = np.load(curr_dict['train']['paths'][0]).shape
        #         print(f'shape {spp} b')
        #     print('sdsd')
        self.data = {}
        keys = ['paths', 'ages'] #'ages' 'handednesses'
        for dict_path in dict_paths:
            with open(dict_path, 'rb') as f:
                curr_dict = pickle.load(f)
                for key in keys:
                    if key in curr_dict[subset]:
                        if key in self.data:
                            self.data[key] = np.concatenate((self.data[key], curr_dict[subset][key]))
                        else:
                            self.data[key] = curr_dict[subset][key]
                    else:
                        emp_arr=np.empty((curr_dict[subset]['paths'].shape[0]))
                        emp_arr[:] = None
                        if key in self.data:
                            self.data[key] = np.concatenate((self.data[key], emp_arr))
                        else:
                            self.data[key] = emp_arr
        
        print('self.data.ages.type,shape', type(self.data['ages']), self.data['ages'].shape)

        # Transforming the age to soft label (probability distribution)
        bin_range = [self.data['ages'].min(), self.data['ages'].max()]
        bin_step = 1
        sigma = 1
        # make soft labels
        for i in range(self.data['ages'].size):
            label = np.array([self.data['ages'][i],]) # Assuming the random subject is 71.3-year-old.
            y, bc = dpu.num2vect(label, bin_range, bin_step, sigma)
            y = np.expand_dims(y,0)
            if i == 0:
                print(f'i = {i}, y.shape {y.shape}')
            if 'labels' in self.data:
                self.data['labels'] = np.concatenate((self.data['labels'], y))
            else:
                self.data['labels'] = y
        print('self.data.labels.type,shape', type(self.data['labels']), self.data['labels'].shape)


    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, index):
        #print(f'dataset getitem subset {self.subset}')
        #image = np.expand_dims(self.brain_volumes[index], axis=0)
        image = np.load(self.data['paths'][index])
        #print(f'image shape {image.shape}')
       # if self.subset == 'train' and self.aug_p > 0:
        #    image = self.monai_transforms(image)
           # print(f'transformed image shape {image.shape}')
       # else:
        #    image = self.test_transforms(image)
            #print(f'test transformed image shape {image.shape}')

        return image, self.data['ages'][index],  #self.data['ages'][index], self.data['handedness'][index], index

class Image_Sex_Dataset(Dataset):
    '''
    This is an example Dataset class that you should adapt to use your own data
    here we produce just random tensors, to illustrate the augmentation pipeline and how it would fit in
    the train_model.py code. For the paper we used UK Biobank data which is freely avaible, but not shareable
    '''

    def __init__(self, subset, aug_p=0.8):
        '''
        Recommended to add oversampling to ensure the generator learns the task properly
        :param aug_p: The augmentation probability. Due to the difficulty of the task and the lack of brain imaging
        available it is recommended that this be set to a fairly high value.
        '''

        self.aug_p = aug_p
        self.subset = subset
        self.test_transforms = t.Compose([t.NormalizeIntensity()])
        self.monai_transforms = [ToTensor(),
                                 t.Rand3DElastic(sigma_range=(0.01, 1), magnitude_range=(0, 1),
                                                 prob=aug_p, rotate_range=(0.18, 0.18, 0.18),
                                                 translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
                                                 spatial_size=None, padding_mode="border", as_tensor_output=False),
                                 t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
                                 t.RandAdjustContrast(prob=aug_p),
                                 t.RandGaussianNoise(prob=aug_p),
                                 t.NormalizeIntensity()]
        self.monai_transforms = t.Compose(self.monai_transforms)
        #'/mnt/disks/bfm/home/ridvan/data/data_paths/hcp/registered/HCP_T1w_1_2_with_age_sex_handedness.pkl', \
        modality = 'T1'
        if modality == 'T1':
            dict_paths = ['/mnt/disks/bfm/home/ridvan/data/data_paths/adni/registered/ADNI_T1w_with_age_sex.pkl']
                        #'/mnt/disks/bfm/home/ridvan/data/data_paths/ppmi/registered/PPMI_T1-anatomical_age_sex_group.pkl',\
                          #  '/mnt/disks/bfm/home/ridvan/data/data_paths/openneuro/registered/openneuro_acq-MPRAGE_T1w_age_sex_handedness.pkl']
        elif modality == 'T2':
            dict_paths = ['/mnt/disks/bfm/home/ridvan/data/data_paths/hcp/registered/HCP_T2w_1_2_with_age_sex_handedness.pkl', \
                        '/mnt/disks/bfm/home/ridvan/data/data_paths/ppmi/registered/PPMI_T2_in_T1-anatomical_space_age_sex_group.pkl',\
                        '/mnt/disks/bfm/home/ridvan/data/data_paths/openneuro/registered/openneuro_acq-CUBE_T2w_age_sex_handedness.pkl']
        else:
            raise Exception('MODALITY ERROR!')
        
        # for dict_path in dict_paths:
        #     with open(dict_path, 'rb') as f:
        #         curr_dict = pickle.load(f)
        #         print(f'dataset {dict_path}')
        #         spp = np.load(curr_dict['train']['paths'][0]).shape
        #         print(f'shape {spp} b')
        #     print('sdsd')
        self.data = {}
        keys = ['paths', 'sexes'] #'ages' 'handednesses'
        for dict_path in dict_paths:
            with open(dict_path, 'rb') as f:
                curr_dict = pickle.load(f)
                for key in keys:
                    if key in curr_dict[subset]:
                        if key in self.data:
                            self.data[key] = np.concatenate((self.data[key], curr_dict[subset][key]))
                        else:
                            self.data[key] = curr_dict[subset][key]
                    else:
                        emp_arr=np.empty((curr_dict[subset]['paths'].shape[0]))
                        emp_arr[:] = None
                        if key in self.data:
                            self.data[key] = np.concatenate((self.data[key], emp_arr))
                        else:
                            self.data[key] = emp_arr


    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, index):
        #print(f'dataset getitem subset {self.subset}')
        #image = np.expand_dims(self.brain_volumes[index], axis=0)
        self.data['paths']
        image = np.load('/mnt/disks/bfm'+self.data['paths'][index])
        #print(f'image shape {image.shape}')
       # if self.subset == 'train' and self.aug_p > 0:
        #    image = self.monai_transforms(image)
           # print(f'transformed image shape {image.shape}')
       # else:
        #    image = self.test_transforms(image)
            #print(f'test transformed image shape {image.shape}')

        return image, self.data['sexes'][index],  #self.data['ages'][index], self.data['handedness'][index], index


class MultiEpochsDataLoader(MonaiDataLoader):
    '''
    Override the default dataloader so that it doesn't spawn loads of processes at the start of every epoch
    This saves a huge amount of time for 3D
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Nicer for tqdm
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.tracker = []

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def track(self):
        """
        @rtype: object
        """
        self.tracker.append(self.avg)

    def save(self, fn):
        np.save(fn, np.array(self.tracker))
