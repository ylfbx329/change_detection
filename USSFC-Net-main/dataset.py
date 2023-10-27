import torch
import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np


def make_dataset(raw_t1, raw_t2, label, test):
    """
    生成数据图片文件地址列表

    :param raw_t1:
    :param raw_t2:
    :param label:
    :param test:
    :return: 包含（t1,t2,label）三张图片路径的元组的列表
    """
    imgs = []
    # 本身是乱序的，file_list[0] = '00000.jpg'
    file_list = os.listdir(raw_t1)
    if test:
        # 按照文件名排序，x = 00000
        file_list.sort(key=lambda x: int(x.split(".")[0]))
    for file in file_list:
        # 拼每个图片的路径
        img_t1 = os.path.join(raw_t1, file)
        img_t2 = os.path.join(raw_t2, file)
        mask = os.path.join(label, file)
        # 包含（t1,t2,label）三张图片路径的元组的列表
        imgs.append((img_t1, img_t2, mask))
    return imgs


class RsDataset(data.Dataset):
    def __init__(self, raw_t1, raw_t2, label, test=False, t1_transform=None, t2_transform=None, label_transform=None):
        # 包含（t1,t2,label）三张图片路径的元组的列表
        imgs = make_dataset(raw_t1, raw_t2, label, test)
        self.imgs = imgs
        self.t1_transform = t1_transform
        self.t2_transform = t2_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        """
        TODO 对分辨率过大的图片裁剪，避免一次性读入内存

        :param index:
        :return img_t1, img_t2, img_y:
        """
        # （t1,t2,label）三张图片路径解包
        t1_path, t2_path, y_path = self.imgs[index]
        img_t1 = Image.open(t1_path)
        img_t2 = Image.open(t2_path)
        img_y = Image.open(y_path)

        # 对三张图片分别进行变换
        if self.t1_transform is not None:
            img_t1 = self.t1_transform(img_t1)
        if self.t2_transform is not None:
            img_t2 = self.t2_transform(img_t2)
        if self.label_transform is not None:
            img_y = self.label_transform(img_y)

        return img_t1, img_t2, img_y

    def __len__(self):
        return len(self.imgs)
