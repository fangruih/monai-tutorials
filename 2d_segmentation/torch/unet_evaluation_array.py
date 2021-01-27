# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader

from monai import config
from monai.data import ArrayDataset, PNGSaver, create_test_image_2d
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, LoadImage, ScaleIntensity, ToTensor


def main(tempdir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_2d(128, 128, num_seg_classes=1)
        Image.fromarray(im.astype("uint8")).save(os.path.join(tempdir, f"img{i:d}.png"))
        Image.fromarray(seg.astype("uint8")).save(os.path.join(tempdir, f"seg{i:d}.png"))

    images = sorted(glob(os.path.join(tempdir, "img*.png")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.png")))

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True), ScaleIntensity(), AddChannel(), ToTensor()])
    segtrans = Compose([LoadImage(image_only=True), AddChannel(), ToTensor()])
    val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    # sliding window inference for one image at every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(torch.load("best_metric_model_segmentation2d_array.pth"))
    model.eval()
    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        saver = PNGSaver(output_dir="./output")
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = post_trans(val_outputs)
            value, _ = dice_metric(y_pred=val_outputs, y=val_labels)
            metric_count += len(value)
            metric_sum += value.item() * len(value)
            saver.save_batch(val_outputs.to(dtype=torch.int))
        metric = metric_sum / metric_count
        print("evaluation metric:", metric)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)