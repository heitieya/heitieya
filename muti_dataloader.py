import os
import torch
import sys
sys.path.append("..")
from dataprocessing import mask
import torchvision
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_dir1, label_dir2, pix, transform=None):
        self.img_dir = img_dir
        self.label_dir1 = label_dir1
        self.label_dir2 = label_dir2
        self.pix = pix

        self.transform = transform
        self.img_labels = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))],
                                 key=lambda x: os.path.getmtime(os.path.join(img_dir, x)))
        self.label_imgs = sorted([f for f in os.listdir(label_dir1) if os.path.isfile(os.path.join(label_dir1, f))],
                                 key=lambda x: os.path.getmtime(os.path.join(label_dir1, x)))
        self.label_coes = sorted([f for f in os.listdir(label_dir2) if os.path.isfile(os.path.join(label_dir2, f))],
                                 key=lambda x: os.path.getmtime(os.path.join(label_dir2, x)))
        self.mask = mask.MatrixModifier((768, 768), (384, 384), 307)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # SFI
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path)
        # wavefront
        label_path1 = os.path.join(self.label_dir1, self.label_imgs[idx])
        label_imgs = Image.open(label_path1)
        # Zernike
        label_path2 = os.path.join(self.label_dir2, self.label_coes[idx])
        label_coes = pd.read_csv(label_path2, header=None)
        label_coes = label_coes.values.tolist()
        label_coes = label_coes[0]
        del label_coes[-1]

        if self.transform:
            image = self.transform(image)
            label_imgs = self.transform(label_imgs)
            label_coes = torch.tensor(label_coes[0:32])
        # mask
        # image = image * torch.tensor(self.mask.modify_matrix(), dtype=torch.float32)
        # down sampling
        image = F.interpolate(image.unsqueeze(0), size=(self.pix, self.pix), mode='bilinear', align_corners=False)
        image = image.squeeze(1)
        label_imgs = F.interpolate(label_imgs.unsqueeze(0), size=(self.pix, self.pix), mode='bilinear', align_corners=False)
        label_imgs = label_imgs.squeeze(1)
        return image, label_imgs, label_coes


# 数据变换
transform = transforms.Compose([
    # transforms.Resize((128, 128)),  # 调整图像大小
    torchvision.transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # 转换为张量

    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# # 创建数据集和数据加载器  real
# img_dir = "/data/lzp/data/wavefront_simulation/train/SFI_mini/"
# label_dir = "/data/lzp/data/wavefront_simulation/train/wavefront_mini/"
# # img_dir = "/data/lzp/data/wavefront_simulation/train/SFI/"
# # label_dir = "/data/lzp/data/wavefront_simulation/train/wavefront/"
# val_img_dir = "/data/lzp/data/low_order_wf/SFI/SFI_val/"
# val_label_dir = "/data/lzp/data/low_order_wf/wfimg_coe/img_val/"
# test_img_dir = "/data/lzp/data/low_order_wf/SFI/SFI_test/"
# test_label_dir = "/data/lzp/data/low_order_wf/wfimg_coe/img_test/"

# # 创建数据集和数据加载器  simulation  normal
# img_dir = "/data/lzp/data/wavefront_simulation/train/SFI_mini/"
# label_dir = "/data/lzp/data/wavefront_simulation/train/wavefront_mini/"
# val_img_dir = "/data/lzp/data/wavefront_simulation/val/SFI/"
# val_label_dir = "/data/lzp/data/wavefront_simulation/val/wavefront/"
# test_img_dir = "/data/lzp/data/wavefront_simulation/test/SFI/"
# test_label_dir = "/data/lzp/data/wavefront_simulation/test/wavefront/"


# 创建数据集和数据加载器  simulation  ketaconus
img_dir = "/data/lzp/data/ketaconus/train/SFI/"
label_dir1 = "/data/lzp/data/ketaconus/train/wf/"
label_dir2 = "/data/lzp/data/ketaconus/train/coe/"

val_img_dir = "/data/lzp/data/ketaconus/val/SFI/"
val_label_dir1 = "/data/lzp/data/ketaconus/val/wf/"
val_label_dir2 = "/data/lzp/data/ketaconus/val/coe/"

test_img_dir = "/data/lzp/data/ketaconus/test/SFI/"
test_label_dir1 = "/data/lzp/data/ketaconus/test/wf/"
test_label_dir2 = "/data/lzp/data/ketaconus/test/coe/"

# # # 创建数据集和数据加载器  simulation  ketaconus
# img_dir = "/data/lzp/data/ketaconus/complex/train/SFI/"
# label_dir1 = "/data/lzp/data/ketaconus/complex/train/wf1/"
# label_dir2 = "/data/lzp/data/ketaconus/complex/train/coe/"
#
# val_img_dir = "/data/lzp/data/ketaconus/complex/val/SFI/"
# val_label_dir1 = "/data/lzp/data/ketaconus/complex/val/wf1/"
# val_label_dir2 = "/data/lzp/data/ketaconus/complex/val/coe/"
#
# test_img_dir = "/data/lzp/data/ketaconus/complex/test/SFI/"
# test_label_dir1 = "/data/lzp/data/ketaconus/complex/test/wf1/"
# test_label_dir2 = "/data/lzp/data/ketaconus/complex/test/coe/"

pix = 256
dataset = CustomImageDataset(img_dir=img_dir, label_dir1=label_dir1, label_dir2=label_dir2, pix=pix,
                             transform=transform)
valdataset = CustomImageDataset(img_dir=val_img_dir, label_dir1=val_label_dir1, label_dir2=val_label_dir2, pix=pix,
                                transform=transform)
testdataset = CustomImageDataset(img_dir=test_img_dir, label_dir1=test_label_dir1,  label_dir2=test_label_dir2, pix=pix,
                                 transform=transform)


# dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
# valdataloader = DataLoader(valdataset, batch_size=5, shuffle=True, num_workers=4)
if __name__ == "__main__":
    i = 0
    # 迭代数据
    for images, labels, l in dataset:
        i = i+1
        print(images.shape, labels.shape, l.shape, i)
