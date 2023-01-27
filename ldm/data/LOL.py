import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import skimage.io
import random


class LOLBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):

        self.gt_dataroot = os.path.join(data_root, 'high')
        self.low_dataroot = os.path.join(data_root, 'low')

        self.gt_image_paths = [f for f in os.listdir(self.gt_dataroot) if '.png' in f]


        self.gt_length = len(self.gt_image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.gt_image_paths],
            "gt_file_path_": [os.path.join(self.gt_dataroot, l) for l in self.gt_image_paths],
            "low_file_path_": [os.path.join(self.low_dataroot, l) for l in self.gt_image_paths]
        }
        self.low_labels = {
            "relative_file_path_": [l for l in self.gt_image_paths]
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        # self.flip = T.Compose([T.RandomHorizontalFlip(p=flip_p),
        #                                 T.RandomVerticalFlip(p=flip_p)])


    def __len__(self):
        return self.gt_length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        gt_image = skimage.io.imread(example["gt_file_path_"])
        low_image = skimage.io.imread(example["low_file_path_"])

        # default to score-sde preprocessing
        gt_img = np.array(gt_image).astype(np.uint8)
        low_img = np.array(low_image).astype(np.uint8)

        crop = min(gt_img.shape[0], gt_img.shape[1])
        h, w, = gt_img.shape[0], gt_img.shape[1]
        gt_img = gt_img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        low_img = low_img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

        gt_image = Image.fromarray(gt_img)
        low_image = Image.fromarray(low_img)
        if self.size is not None:
            gt_image = gt_image.resize((self.size, self.size), resample=self.interpolation)
            low_image = low_image.resize((self.size, self.size), resample=self.interpolation)

        gt_image  = np.array(gt_image).astype(np.uint8)
        low_image = np.array(low_image).astype(np.uint8)

        img = np.concatenate([gt_image, low_image], axis=-1)

        hflip = random.random()
        vflip = random.random()

        if hflip < self.flip_p:
            img = np.flip(img, axis=0)
        if vflip < self.flip_p:
            img = np.flip(img, axis=1)

        example["image"] = (img[:, :, 0:3] / 127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (img[:, :, 3:6] / 127.5 - 1.0).astype(np.float32)
        return example


class LOLTrain(LOLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LOLValidation(LOLBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
