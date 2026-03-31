# """
#  @Time    : 2021/7/6 10:56
#  @Author  : Haiyang Mei
#  @E-mail  : mhy666@mail.dlut.edu.cn
 
#  @Project : CVPR2021_PFNet
#  @File    : datasets.py
#  @Function: Datasets Processing
 
# """
# import os
# import os.path
# import torch.utils.data as data
# from PIL import Image

# def make_dataset(root):
#     image_path = os.path.join(root, 'image')
#     mask_path = os.path.join(root, 'mask')
#     img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
#     return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

# class ImageFolder(data.Dataset):
#     # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
#     def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
#         self.root = root
#         self.imgs = make_dataset(root)
#         self.joint_transform = joint_transform
#         self.transform = transform
#         self.target_transform = target_transform

#     def __getitem__(self, index):
#         img_path, gt_path = self.imgs[index]
#         img = Image.open(img_path).convert('RGB')
#         target = Image.open(gt_path).convert('L')
#         if self.joint_transform is not None:
#             img, target = self.joint_transform(img, target)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)

import glob, random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class HillshadeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None,
                 road_biased=True, road_ratio=0.70, road_min_pixels=30):

        self.img_files  = sorted(glob.glob(f"{img_dir}/*.npy"))
        self.mask_files = sorted(glob.glob(f"{mask_dir}/*.npy"))
        self.transform  = transform

        self.road_files, self.bg_files = [], []

        if road_biased:
            for img_f, mask_f in zip(self.img_files, self.mask_files):
                mask = np.load(mask_f)
                if (mask.squeeze() > 0.5).sum() >= road_min_pixels:
                    self.road_files.append((img_f, mask_f))
                else:
                    self.bg_files.append((img_f, mask_f))

        self.road_biased = road_biased
        self.road_ratio  = road_ratio
        self._all = list(zip(self.img_files, self.mask_files))

    def __len__(self):
        return len(self.img_files)

    def _load(self, img_path, mask_path):
        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if img.shape[0] == 4:
            img = img.transpose(1, 2, 0)

        mask = (mask > 0.5).astype(np.float32)
        return img, mask

    def __getitem__(self, idx):
        if self.road_biased and self.road_files:
            pool = self.road_files if random.random() < self.road_ratio else self.bg_files
            img_p, mask_p = random.choice(pool)
        else:
            img_p, mask_p = self._all[idx]

        img, mask = self._load(img_p, mask_p)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, mask.unsqueeze(0)



import albumentations as A
from albumentations.pytorch import ToTensorV2
# from configs.config import CFG

# def get_transforms(phase):
#     if phase == "train":
#         return A.Compose([
#             A.RandomCrop(CFG.PATCH_SIZE, CFG.PATCH_SIZE),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.GaussNoise(p=0.3),
#             A.RandomBrightnessContrast(p=0.3),
#             ToTensorV2(transpose_mask=False),
#         ])
#     else:
#         return A.Compose([
#             A.CenterCrop(CFG.PATCH_SIZE, CFG.PATCH_SIZE),
#             ToTensorV2(transpose_mask=False),
#         ])


def get_transforms(phase, patch_size):
    if phase == "train":
        return A.Compose([
            A.RandomCrop(patch_size, patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.5]*4, std=[0.5]*4),
            ToTensorV2(transpose_mask=False),
        ])
    else:
        return A.Compose([
            A.CenterCrop(CFG.PATCH_SIZE, CFG.PATCH_SIZE),
            A.Normalize(mean=[0.5]*4, std=[0.5]*4),
            ToTensorV2(transpose_mask=False),
        ])