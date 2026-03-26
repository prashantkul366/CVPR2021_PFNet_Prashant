"""
 @Time    : 2021/7/6 10:56
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : datasets.py
 @Function: Datasets Processing
 
"""
import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch

# def make_dataset(root):
#     image_path = os.path.join(root, 'image')
#     mask_path = os.path.join(root, 'mask')
#     img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
#     return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

def make_dataset(root):

    image_path = os.path.join(root, 'images')
    mask_path = os.path.join(root, 'masks')

    # img_list = [f for f in os.listdir(image_path) if f.endswith('.png')]
    img_list = sorted([f for f in os.listdir(image_path) if f.endswith('.npy')])

    return [
        (os.path.join(image_path, f),
         os.path.join(mask_path, f))
        for f in img_list
    ]

class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img_path, gt_path = self.imgs[index]   # ⭐ FIX

        img = np.load(img_path).astype(np.float32)   # H W 4
        mask = np.load(gt_path).astype(np.float32)

        # ---------- FIX SHAPE ----------
        if img.shape[0] == 4:
            img = np.transpose(img, (1,2,0))   # C,H,W → H,W,C

        # ---------- CHANNEL NORMALIZATION ----------
        for c in range(img.shape[2]):
            img[:,:,c] = (img[:,:,c] - img[:,:,c].mean()) / (img[:,:,c].std() + 1e-6)

        # ---------- TO TENSOR ----------
        img = torch.from_numpy(img).permute(2,0,1).float()   # 4,H,W
        mask = torch.from_numpy(mask).unsqueeze(0).float()   # 1,H,W

        # ---------- BINARIZE ----------
        mask = (mask > 0).float()

        # ---------- RESIZE (VERY IMPORTANT) ----------
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(416,416), mode='bilinear', align_corners=False
        ).squeeze(0)

        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(416,416), mode='nearest'
        ).squeeze(0)

        return img, mask
    
    def __len__(self):
        return len(self.imgs)


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
# import numpy as np
# import torch

# def make_dataset(root):
#     # image_path = os.path.join(root, 'image')
#     # mask_path = os.path.join(root, 'mask')
#     image_path = os.path.join(root, 'images')
#     mask_path = os.path.join(root, 'masks')
#     # img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
#     # return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]
#     # img_list = [f for f in os.listdir(image_path) if f.endswith('.npy')]
#     img_list = sorted([f for f in os.listdir(image_path) if f.endswith('.npy')])

#     return [(os.path.join(image_path, f),
#             os.path.join(mask_path, f)) for f in img_list]

# class ImageFolder(data.Dataset):
#     # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
#     def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
#         self.root = root
#         self.imgs = make_dataset(root)
#         self.joint_transform = joint_transform
#         self.transform = transform
#         self.target_transform = target_transform

#     # def __getitem__(self, index):
#     #     img_path, gt_path = self.imgs[index]
#     #     # img = Image.open(img_path).convert('RGB')
#     #     # target = Image.open(gt_path).convert('L')
#     #     img = np.load(img_path)      # (H,W,4)
        
#     #     target = np.load(gt_path)    # (H,W)

#     #     # if self.joint_transform is not None:
#     #     #     img, target = self.joint_transform(img, target)
#     #     # if self.transform is not None:
#     #     #     img = self.transform(img)
#     #     # if self.target_transform is not None:
#     #     #     target = self.target_transform(target)

#     #     # return img, target
#     #     # to tensor
#     #     img = torch.from_numpy(img).float()
#     #     img = img.permute(1, 0, 2) 

#     #     target = torch.from_numpy(target).float()
#     #     target = target.unsqueeze(0) # -> 1 H W

#     #     return img, target
#     def __getitem__(self, index):

#         img_path, gt_path = self.imgs[index]
        
#         img = np.load(img_path)

#         img = torch.from_numpy(img).float()

#         # ===== FIX CHANNEL POSITION =====
#         if img.shape[0] == 4:
#             # already C H W
#             pass
#         elif img.shape[1] == 4:
#             # H C W → C H W
#             img = img.permute(1, 0, 2)
#         elif img.shape[2] == 4:
#             # H W C → C H W
#             img = img.permute(2, 0, 1)
#         else:
#             raise ValueError(f"Unknown image shape {img.shape}")

#         # ===== NORMALIZE =====
#         for c in range(img.shape[0]):
#             img[c] = (img[c] - img[c].mean()) / (img[c].std() + 1e-6)
            
#         # img = np.load(img_path)      # (H,W,4)
#         target = np.load(gt_path)    # (H,W)

#         # img = torch.from_numpy(img).float()
#         # img = img.permute(2, 0, 1)   

#         target = torch.from_numpy(target).float()
#         target = (target > 0).float()   # ⭐ BINARIZE
#         target = target.unsqueeze(0)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)
