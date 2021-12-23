import os
import cv2
import torch
import random
import shutil
import numpy as np
from torch import nn
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class SelfDatasetFolder(VisionDataset):

    def __init__(self, imgroot, transform=None):
        super(SelfDatasetFolder, self).__init__(imgroot, transform=transform)
        samples = self.make_dataset(imgroot)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.samples = samples
        # self.targets = [s[1] for s in samples]

    def make_dataset(self, imgroot):
        instances = []
        target_imgroot = os.path.join(imgroot, "moban")
        image_imgroot = os.path.join(imgroot, "shuru")
        for r, dir, files in os.walk(image_imgroot):
            # print("len(files)", len(files))
            for file in files:
                fn, ext = os.path.splitext(file)
                if ext not in IMG_EXTENSIONS:
                    continue
                target_name = os.path.join(target_imgroot, file)
                image_name = os.path.join(image_imgroot, file)
                if os.path.exists(target_name):
                    img = cv2.imread(os.path.join(r, file))
                    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([img], [1], None, [256], [0, 256])
                    hist3 = cv2.calcHist([img], [2], None, [256], [0, 256])
                    # print(type(hist1))
                    number1, number2, number3 = hist1.sum(), hist2.sum(), hist3.sum()
                    bn_hist1, bn_hist2, bn_hist3 = np.zeros_like(hist1), np.zeros_like(hist1), np.zeros_like(hist1)
                    for i in range(1):
                        for j in range(256):
                            bn_hist1[j][i] = hist1[j][i] / number1
                            bn_hist2[j][i] = hist2[j][i] / number2
                            bn_hist3[j][i] = hist3[j][i] / number3
                    target = np.stack((bn_hist1, bn_hist2, bn_hist3))
                    target = np.squeeze(target)
                    # target = np.concatenate((bn_hist1, bn_hist2, bn_hist3))
                    # target = target.reshape((3, 256))

                    # target = transforms.ToTensor()(target)
                    # target = torch.unsqueeze(target, dim=2)
                    # bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)
                    # target = bn(target)
                item = image_name, target
                instances.append(item)
        return instances

    def loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except:
            print("Readimg error !!!")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path, target

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    imgroot = r'D:\Datasets\liaoning\train'
    dataset = SelfDatasetFolder(imgroot)
    # print("data num: ", len(dataset))
    # sample, path = dataset[0]
    # print(sample, path)
    # sample, path = dataset[-1]
    # print(sample, path)
    # instances = []
    # for r, dir, files in os.walk(r'D:\Datasets\yunse\val'):
    #     print(files)
    #     for file in files:
    #         #     print(file)
    #         #     fn, ext = os.path.splitext(file)
    #         #     if ext not in IMG_EXTENSIONS:
    #         #         continue
    #         item = os.path.join(r, file)
    #         instances.append(item)
