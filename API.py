import os
import cv2
import time
import torch


path = r'D:\Wasserstein\checkpoint.pth'
checkpoint = torch.jit.load(path)
print(checkpoint)