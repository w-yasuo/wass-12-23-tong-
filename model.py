from typing import Any
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.init as init
from torch.nn import functional as F


# from torchvision.models.utils import load_state_dict_from_url


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2)
        self.conv1_relu = nn.ReLU()
        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        # self.input_pooling = nn.MaxPool2d(kernel_size=16, stride=8)
        # self.conv1_pooling = nn.MaxPool2d(kernel_size=4, stride=4)
        # self.fire3_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fire5_identity = nn.Identity()
        self.conv10_deconv = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)

        self._fc1 = nn.Linear(725355, 256)
        self._fc2 = nn.Linear(725355, 256)
        self._fc3 = nn.Linear(725355, 256)

        self.Softmax = nn.Softmax(dim=1)

    def extract_features(self, input):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """

        #     convolution     #
        conv1 = self.conv1(input)
        bn = nn.BatchNorm2d(96)(conv1)
        relu = self.conv1_relu(bn)
        fire2 = self.fire2(nn.MaxPool2d(kernel_size=3, stride=2)(relu))
        fire3 = self.fire3(fire2)
        fire4 = self.fire4(nn.MaxPool2d(kernel_size=3, stride=2)(fire3))
        fire5 = self.fire5(fire4)
        fire6 = self.fire6(nn.MaxPool2d(kernel_size=3, stride=2)(fire5))
        fire7 = self.fire7(fire6)
        fire8 = self.fire8(fire7)
        fire9 = self.fire9(fire8)
        conv10 = self.conv10(fire9)
        # print("conv1:", conv1.shape)        # torch.Size([1, 96, 111, 111])
        # print("relu:", relu.shape)          # torch.Size([1, 96, 111, 111])
        # # print("maxpool1:", maxpool1.shape)  # torch.Size([1, 96, 55, 55])
        # print("fire2:", fire2.shape)        # torch.Size([1, 128, 55, 55])
        # print("fire3:", fire3.shape)        # torch.Size([1, 128, 55, 55])
        # print("maxpool3:", maxpool3.shape)  # torch.Size([1, 128, 27, 27])
        # print("fire4:", fire4.shape)        # torch.Size([1, 256, 27, 27])
        # print("fire5:", fire5.shape)        # torch.Size([1, 256, 27, 27])
        # # print("maxpool5:", maxpool5.shape)  # torch.Size([1, 256, 13, 13])
        # print("fire6:", fire6.shape)        # torch.Size([1, 384, 13, 13])
        # print("fire7:", fire7.shape)        # torch.Size([1, 384, 13, 13])
        # print("fire8:", fire8.shape)        # torch.Size([1, 512, 13, 13])
        # print("fire9:", fire9.shape)        # torch.Size([1, 512, 13, 13])
        # print("conv10:", conv10.shape)      # torch.Size([1, 512, 13, 13])
        #     downsampling      #
        input_pooling = nn.MaxPool2d(kernel_size=16, stride=8)(input)
        conv1_pooling = nn.MaxPool2d(kernel_size=4, stride=4)(self.conv1(input))
        fire3_pooling = nn.MaxPool2d(kernel_size=2, stride=2)(fire3)
        fire5_pooling = nn.Identity()(fire5)
        conv10_deconv = self.conv10_deconv(conv10)
        # print(input_pooling.shape)
        # print(conv1_pooling.shape)
        # print(fire3_pooling.shape)
        # print(fire5_pooling.shape)
        # print(conv10_deconv.shape)

        #     concatenate & flatten     #
        input_pooling_flatten = torch.flatten(input_pooling, 1)  # torch.Size([2187])
        conv1_pooling_flatten = torch.flatten(conv1_pooling, 1)  # torch.Size([69984])
        fire3_pooling_flatten = torch.flatten(fire3_pooling, 1)  # torch.Size([93312])
        fire5_pooling_flatten = torch.flatten(fire5_pooling, 1)  # torch.Size([186624])
        conv10_deconv_flatten = torch.flatten(conv10_deconv, 1)  # torch.Size([373248])
        # print(input_pooling_flatten.detach().numpy().resize(27,27,3))

        fina_x = np.concatenate((input_pooling_flatten.detach().numpy(), conv1_pooling_flatten.detach().numpy(),
                                 fire3_pooling_flatten.detach().numpy(), fire5_pooling_flatten.detach().numpy(),
                                 conv10_deconv_flatten.detach().numpy()), axis=-1)  # print(fina_x.shape)  # (725355,)

        x = torch.tensor(fina_x)
        # print("x.shape:", x.shape)  # torch.Size([92845440])
        fc1 = self._fc1(x)
        fc2 = self._fc2(x)
        fc3 = self._fc3(x)
        # print("fc1.shape:", fc1.shape)
        # print(fc1[0].shape)
        drop1 = nn.Dropout(p=0.5)(fc1)
        drop2 = nn.Dropout(p=0.5)(fc2)
        drop3 = nn.Dropout(p=0.5)(fc3)
        # return fc1, fc2, fc3
        return drop1, drop2, drop3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = self.extract_features(x)
        S_x1 = self.Softmax(x1)
        S_x2 = self.Softmax(x2)
        S_x3 = self.Softmax(x3)
        # pred = np.concatenate((S_x1.detach().numpy(), S_x2.detach().numpy(), S_x3.detach().numpy()))
        # pred = pred.reshape((-1, 3, 256))  # (24, 3, 256)
        pred = np.stack((S_x1.detach().numpy(), S_x2.detach().numpy(), S_x3.detach().numpy()))



        pred = transforms.ToTensor()(pred)
        # pred = torch.unsqueeze(pred, dim=2)
        return pred


if __name__ == '__main__':
    # 加载模型
    model = SqueezeNet()
    # 获取图片
    image1 = Image.open(r"D:\Datasets\yunse\train\crop_img_1_34.tif")
    image2 = Image.open(r"D:\Datasets\yunse\train\crop_img_31_12.tif")
    image3 = Image.open(r"D:\Datasets\yunse\train\crop_img_10_33.tif")
    image4 = Image.open(r"D:\Datasets\yunse\train\crop_img_10_32.tif")
    image_a = np.concatenate(
        (np.array(image1.resize((224, 224))), np.array(image2.resize((224, 224))), np.array(image3.resize((224, 224))), np.array(image4.resize((224, 224))),),
        axis=0)
    image = Image.fromarray(np.array(image_a))
    toTensor = transforms.ToTensor()  # 实例化一个toTensor
    image_tensor = toTensor(image)
    image_tensor = image_tensor.reshape(4, 3, 224, 224)
    output1 = model(image_tensor)
    # print(output1, output2, output3)
    print("out", output1.shape)
    # # print("model:", model)

    # 获取图片
    # img = cv2.imread(r"D:\TEST_IMAGE\2.jpg")
    # img= cv2.resize(img, (224, 224))
    # toTensor = transforms.ToTensor()
    # img = toTensor(img)
    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img)
    # output1, output2, output3 = model(img)
    # print("output1:", output1)
    # print("output2:", output2)
    # print("output3:", output3)
