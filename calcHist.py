import os
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from model import SqueezeNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def plot_demo(image):
    # numpy的ravel函数功能是将多维数组降为一维数组
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()
color = {"blue", "green", "red"}


def image_hist_demo(image):
    color = {"blue", "green", "red"}
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。
    for i, color in enumerate(color):
        hist1 = cv2.calcHist([image], [i], None, [256], [0, 256])
        T_hist_B = np.expand_dims(hist1, 0)
        T_hist_B = np.expand_dims(T_hist_B, 0)
        T_hist_B = torch.tensor(T_hist_B)
        hist = bn(T_hist_B)

        hist = torch.squeeze(hist, 0)
        hist = torch.squeeze(hist, 0)

        plt.figure(1, figsize=(15, 10))
        plt.subplot(1, 3, i + 1)
        plt.plot(hist, color=color)
        plt.figure(2, figsize=(15, 10))
        plt.plot(hist1, color=color)
        plt.xlim([0, 256])
    plt.show()


if __name__ == "__main__":
    bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)
    totensor = transforms.ToTensor()
    loss = nn.MSELoss()
    img = cv2.imread(r"D:\TEST_IMAGE\2.jpg")
    # cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("input image", img)
    # plot_demo(img)
    # image_hist_demo(img)

    model = SqueezeNet()
    image = cv2.resize(img, (224, 224))
    toTensor = transforms.ToTensor()  # 实例化一个toTensor
    image_tensor = toTensor(image)
    image_tensor = image_tensor.reshape(1, 3, 224, 224)
    output1, output2, output3 = model(image_tensor)

    for i, color in enumerate(color):
        hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
        T_hist_B = np.expand_dims(hist1, 0)
        T_hist_B = np.expand_dims(T_hist_B, 0)
        print(T_hist_B.shape)
        T_hist_B = torch.tensor(T_hist_B)
        print(T_hist_B.shape)
        hist = bn(T_hist_B)

        # hist = torch.squeeze(hist, 0)
        # hist = torch.squeeze(hist, 0)
        hist = torch.flatten(hist, 0, 2)
        plt.figure(3, figsize=(15, 10))
        # plt.subplot(1, 3, i+1)
        plt.plot(hist, color=color)
    plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 显示三通道的直方图
    # hist_B = cv2.calcHist([img], [0], None, [256], [0, 256])
    # hist_G = cv2.calcHist([img], [1], None, [256], [0, 256])
    # hist_R = cv2.calcHist([img], [2], None, [256], [0, 256])
    #
    # T_hist_B = np.expand_dims(hist_B, 0)
    # T_hist_B = np.expand_dims(T_hist_B, 0)
    # T_hist_B = torch.tensor(T_hist_B)
    # S_T_hist_B = nn.Softmax(dim=2)(T_hist_B)
    # BN_T_hist_B = bn(T_hist_B)
    #
    # T_hist_G = np.expand_dims(hist_G, 0)
    # T_hist_G = np.expand_dims(T_hist_G, 0)
    # T_hist_G = torch.tensor(T_hist_G)
    # S_T_hist_G = nn.Softmax(dim=2)(T_hist_G)
    # BN_T_hist_G = bn(T_hist_G)
    #
    # T_hist_R = np.expand_dims(hist_R, 0)
    # T_hist_R = np.expand_dims(T_hist_R, 0)
    # T_hist_R = torch.tensor(T_hist_R)
    # S_T_hist_R = nn.Softmax(dim=2)(T_hist_R)
    # BN_T_hist_R = bn(T_hist_R)
    #
    # S_B_G = loss(S_T_hist_B, S_T_hist_G)
    # s_B_G = loss(BN_T_hist_B, BN_T_hist_G)
    #
    # S_B_R = loss(S_T_hist_B, S_T_hist_R)
    # s_B_R = loss(BN_T_hist_B, BN_T_hist_R)
    #
    # S_G_R = loss(S_T_hist_G, S_T_hist_R)
    # s_G_R = loss(BN_T_hist_G, BN_T_hist_R)
    # print(S_B_G, s_B_G)
    # print(S_B_R, s_B_R)
    # print(S_G_R, s_G_R)
