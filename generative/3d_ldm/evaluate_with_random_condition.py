# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from generative.inferers import LatentDiffusionInferer
# from generative.networks.schedulers import DDPMScheduler
from generative.networks.schedulers import DDIMScheduler as monai_ddimscheduler
from monai.config import print_config
from monai.utils import set_determinism

from utils import define_instance
from util.training_utils import generate_random_condition

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Latent Diffusion Model Inference")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="number of generated images",
    )
    args = parser.parse_args()

    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    
    return args
def evaluate_with_random_condition(args, diffusion_evaluation_checkpoint):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(42)
    


    # load trained networks
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(args.autoencoder_dir, "autoencoder.pt")
    autoencoder.load_state_dict(torch.load(trained_g_path))

    diffusion_model = define_instance(args, "diffusion_def").to(device)
    # trained_diffusion_path = os.path.join(args.diffusion_dir, "diffusion_unet.pt")
    # diffusion_model.load_state_dict(torch.load(trained_diffusion_path))
    diffusion_model.load_state_dict(diffusion_evaluation_checkpoint)

    # scheduler = DDPMScheduler(
    #     num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
    #     schedule="scaled_linear_beta",
    #     beta_start=args.NoiseScheduler["beta_start"],
    #     beta_end=args.NoiseScheduler["beta_end"],
    # )
    scheduler = monai_ddimscheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        # schedule="scaled_linear_beta",
        # beta_start=args.NoiseScheduler["beta_start"],
        # beta_end=args.NoiseScheduler["beta_end"],
    )
    scheduler.set_timesteps(50)
    
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)
    
    
    args.evaluation_output_dir= os.path.join(args.evaluation_output_dir,"direct_conditioned_generation", common_timestamp)
    
    Path(args.evaluation_output_dir).mkdir(parents=True, exist_ok=True)
    latent_shape = [p // 4 for p in args.diffusion_train["patch_size"]]
    noise_shape = [1, args.latent_channels] + latent_shape

    for _ in range(args.num):
        noise = torch.randn(noise_shape, dtype=torch.float32).to(device)
        # noise = torch.randint(low=0, high=1024, size=noise_shape)
        random_condition, age, sex = generate_random_condition(device, expand=True, cross_attention_dim=250, expand_token_times=100)
        random_condition = random_condition.unsqueeze(0)
        print("random_condition, age, sex", random_condition, age, sex)
        print("random_condition", random_condition.shape)
        with torch.no_grad():
            # synthetic_images = inferer.sample(
            #     input_noise=noise,
            #     autoencoder_model=autoencoder,
            #     diffusion_model=diffusion_model,
            #     scheduler=scheduler,
            #     conditioning=None, 
            #     # conditioning=torch.tensor([[[138.,   1.,   0.]]]).to(device), 
            # )
            synthetic_images = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                conditioning=random_condition, 
                # conditioning=torch.tensor([[[138.,   1.,   0.]]]).to(device), 
            )
            
        filename = os.path.join(args.output_dir, f"synimg_age_{int(age)}_sex_{sex}.nii.gz")
        # filename = os.path.join(args.output_dir, datetime.now().strftime("synimg_%Y%m%d_%H%M%S"))
        final_img = nib.Nifti1Image(synthetic_images[0, 0, ...].unsqueeze(-1).cpu().numpy(), np.eye(4))
        nib.save(final_img, filename)
    return args.evaluation_output_dir


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments()    
    evaluate_with_random_condition(args)
    
    
