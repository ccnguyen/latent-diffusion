import cv2
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import skimage.io


def decrease_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    v = v / 1.0
    v *= value
    v = v.astype(np.uint8)
    # v[v < value] = 0
    # v[v >= value] -= value
    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img


class CustomDataset(Dataset):
    def __init__(self, data_root, out_root, size, bright, noise):
        self.bright = bright
        self.noise = noise

        self.dir = data_root
        dataset = data_root.split('/')[-1]
        self.dataset_name = dataset
        self.output_dir = f'{out_root}/{self.dataset_name}_png_v{self.bright}_n{self.noise}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.files = [f for f in os.listdir(self.dir) if '.png' in f]
        self.labels = {
            "relative_file_path_": sorted([f for f in os.listdir(self.dir) if '.png' in f]),
            "gt_file_path_": sorted([f for f in os.listdir(self.dir) if '.png' in f]),
            "low_file_path_": sorted([f for f in os.listdir(self.dir) if '.png' in f])
        }


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        example = dict((k, self.labels[k][idx]) for k in self.labels)
        example['dir'] = self.output_dir

        og_img = skimage.io.imread(f'{self.dir}/{example["gt_file_path_"]}')
        img = decrease_brightness(og_img, value=self.bright) / 255.0
        img = img + np.random.randn(*img.shape) * self.noise
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)[None, ...]
        og_img = torch.from_numpy(og_img / 255.0).permute(2, 0, 1)[None,...]

        og_h = img.shape[-2]
        og_w = img.shape[-1]
        og_img = F.interpolate(og_img, size=(256, 256), align_corners=False, mode='bilinear', antialias=True)

        img = F.interpolate(img, size=(256, 256), align_corners=False, mode='bilinear', antialias=True)

        img = img[0].permute(1,2,0).numpy()
        og_img = og_img[0].permute(1,2,0).numpy()
        example['image'] = og_img * 2 - 1
        example['LR_image'] = img * 2 - 1
        example['og_h'] = og_h
        example['og_w'] = og_w

        return example

class DIV2k(Dataset):
     def __init__(self, data_root, out_root, size, bright, noise):
         self.bright = bright
         self.noise = noise

         self.dir = data_root
         dataset = data_root.split('/')[-1]
         self.dataset_name = dataset
         self.output_dir = f'{out_root}'
         os.makedirs(self.output_dir, exist_ok=True)
         self.files = [f for f in os.listdir(self.dir) if '.png' in f]
         self.labels = {
             "relative_file_path_": sorted([f for f in os.listdir(self.dir) if '.png' in f]),
             "gt_file_path_": sorted([f for f in os.listdir(self.dir) if '.png' in f]),
             "low_file_path_": sorted([f for f in os.listdir(self.dir) if '.png' in f])
         }


     def __len__(self):
         return len(self.files)

     def __getitem__(self, idx):
         example = dict((k, self.labels[k][idx]) for k in self.labels)
         example['dir'] = self.output_dir

         og_img = skimage.io.imread(f'{self.dir}/{example["gt_file_path_"]}') / 255.0
         # img = decrease_brightness(og_img, value=self.bright) / 255.0
         # img = img + np.random.randn(*img.shape) * self.noise
         # img = np.clip(img, 0.0, 1.0).astype(np.float32)
         img = torch.from_numpy(og_img.copy()).permute(2, 0, 1)[None, ...]
         og_img = torch.from_numpy(og_img / 255.0).permute(2, 0, 1)[None,...]

         og_h = img.shape[-2]
         og_w = img.shape[-1]
         og_img = F.interpolate(og_img, size=(256, 256), align_corners=False, mode='bilinear', antialias=True)

         img = F.interpolate(img, size=(256, 256), align_corners=False, mode='bilinear', antialias=True)

         img = img[0].permute(1,2,0).numpy()
         og_img = og_img[0].permute(1,2,0).numpy()
         example['image'] = og_img * 2 - 1
         example['LR_image'] = img * 2 - 1
         example['og_h'] = og_h
         example['og_w'] = og_w

         return example

class CustomTrain(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CustomValidation(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)