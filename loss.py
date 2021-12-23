import os
import cv2
import torch
import numpy as np
from torch import nn
from scipy import *


# def L2loss(target, pred):
#     loss = 0
#     # for number in range(target.size(0)):
#     for number in range(len(target)):
#         for i in range(3):
#             h1 = np.zeros((224 * 224))
#             h2 = np.zeros((224 * 224))
#             target[number][i, :, 0] = np.cumsum(target[number][i])
#             pred[number][i, :, 0] = np.cumsum(pred[number][i])
#             for j in range(len(target[number][i])):
#                 if j == 0:
#                     h1[:int(target[number][i, j, 0])] = 0
#                     h2[:int(pred[number][i, j, 0])] = 0
#                 elif j == len(target[number][i]) - 1:
#                     h1[int(target[number][i, j, 0]):] = j
#                     h2[int(pred[number][i, j, 0]):] = j
#                 else:
#                     h1[int(target[number][i, j - 1, 0]):int(target[number][i, j, 0])] = j
#                     h2[int(pred[number][i, j - 1, 0]):int(pred[number][i, j, 0])] = j
#
#             loss += np.sqrt(np.sum(np.square(np.absolute(h1 - h2))))
#     return loss
def L2loss(target, pred):
    # print(target.shape, pred.shape)  # [48, 3, 256] [256, 3, 48]
    pred = pred.permute(2, 1, 0)
    loss = 0
    for batch in range(len(target)):
        # print(len(target[batch]))
        # print(len(pred[batch]))
        for c in range(3):
            target_his = target[batch][c]
            pred_his = pred[batch][c]
            # print(target[batch][c].shape)
            # print(pred[batch][c].shape)

            abs = np.absolute(target_his - pred_his)
            squ = np.square(abs)
            squ_arr = np.array(squ)
            sum = np.sum(squ_arr)
            sqrt = np.sqrt(sum)
            loss += sqrt
    return loss

    # loss = 0
    # sum, sqrt = 0, 0
    # for batch in range(len(target)):  # torch.Size([48, 3, 256])
    #     T = torch.squeeze(torch.tensor(target[batch]))
    #     P = torch.squeeze(torch.tensor(pred[batch]))
    #     print(P.shape)
    #     print(T.shape)
    #     abs = np.absolute(T - P)
    #     squ = np.square(abs)
    #     squ_arr = np.array(squ)
    #     sum = np.sum(squ_arr)
    #     sqrt = np.sqrt(sum)
    #     loss += sqrt
    # print(sum, sqrt, loss)
    # quit()
        # for c in range(3):
        #     # print(target[batch])
        #     print(target[batch][c].shape)
        #     # print(pred[batch])
        #     print(pred[batch][c].shape)
        #     quit()
        #     target_his = target[batch][c]
        #     pred_his = pred[batch][c]
        #     # print("P", pred_his.shape)  # P torch.Size([256])
        #     # print("T", target_his.shape)  # T torch.Size([256])
        #     try:
        #         abs = np.absolute(target_his - pred_his)
        #         squ = np.square(abs)
        #         squ_arr = np.array(squ)
        #         sum = np.sum(squ_arr)
        #         sqrt = np.sqrt(sum)
        #         loss += sqrt
        #     except:
        #         print("P", pred_his.shape)  # torch.Size([3])
        #         print("T", target_his.shape)  # torch.Size([256])

                # print(target_his)
                # print(pred_his)

            # loss += np.sqrt(np.sum(np.array(np.square(np.absolute(target_his - pred_his)))))
    # return loss

# # bs = 1
# def L2loss(target, pred):
#     loss = 0
#     for i in range(3):
#         h1 = np.zeros((224 * 224))
#         h2 = np.zeros((224 * 224))
#         target[i, :, 0] = np.cumsum(target[i])
#         pred[i, :, 0] = np.cumsum(pred[i])
#
#         for j in range(len(target[i])):
#             if j == 0:
#                 # print(target[i, j, 0])
#                 h1[:int(target[i, j, 0])] = 0
#                 h2[:int(pred[i, j, 0])] = 0
#             elif j == len(target[i]) - 1:
#                 h1[int(target[i, j, 0]):] = j
#                 h2[int(pred[i, j, 0]):] = j
#             else:
#                 h1[int(target[i, j - 1, 0]):int(target[i, j, 0])] = j
#                 h2[int(pred[i, j - 1, 0]):int(pred[i, j, 0])] = j
#         loss += np.sqrt(np.sum(np.square(np.absolute(h1 - h2))))
#     return loss

if __name__ == '__main__':
    # img1 = cv2.imread(r"D:\TEST_IMAGE\3.jpg")
    # img1 = cv2.imread(r"D:\Datasets\yunse\train\crop_img_26_52.tif")
    # target_list = []
    # for r, dir, files in os.walk(r'D:\Datasets\test\train'):
    #     for file in files:
    #         fn, ext = os.path.splitext(file)
    #         if ext not in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
    #             continue
    #         target_name = os.path.join(os.path.join(r'D:\Datasets\test\train', "moban"), file)
    #         image_name = os.path.join(os.path.join(r"D:\Datasets\test\train", "shuru"), file)
    #         if os.path.exists(target_name):
    #             img = cv2.imread(os.path.join(r, file))
    #             hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    #             hist2 = cv2.calcHist([img], [1], None, [256], [0, 256])
    #             hist3 = cv2.calcHist([img], [2], None, [256], [0, 256])
    #             target = np.concatenate((hist1, hist2, hist3))
    #             target = target.reshape((3, 256, 1))
    #             # target = transforms.ToTensor()(target)
    #             # target = torch.unsqueeze(target, dim=2)
    #             # bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)
    #             # target = bn(target)
    #         # item = target
    #         target_list.append(target)
    # pred_list = []
    # for r, dir, files in os.walk(r'D:\Datasets\test\train'):
    #     print(files)
    #     for file in files:
    #         fn, ext = os.path.splitext(file)
    #         if ext not in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
    #             continue
    #         target_name = os.path.join(os.path.join(r'D:\Datasets\test\train', "moban"), file)
    #         image_name = os.path.join(os.path.join(r"D:\Datasets\test\train", "shuru"), file)
    #         if os.path.exists(image_name):
    #             img = cv2.imread(os.path.join(r, file))
    #             hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    #             hist2 = cv2.calcHist([img], [1], None, [256], [0, 256])
    #             hist3 = cv2.calcHist([img], [2], None, [256], [0, 256])
    #             pred = np.concatenate((hist1, hist2, hist3))
    #             pred = target.reshape((3, 256, 1))
    #             # target = transforms.ToTensor()(target)
    #             # target = torch.unsqueeze(target, dim=2)
    #             # bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)
    #             # target = bn(target)
    #         # item = target
    #         pred_list.append(pred)
    # loss = L2loss(target_list, pred_list)
    # print(loss)

    img1 = cv2.imread(r"D:\Datasets\test\train\moban\crop_7_3.jpg")
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    hist3 = cv2.calcHist([img1], [2], None, [256], [0, 256])
    # pred = np.concatenate((hist1, hist2, hist3))
    pred = np.stack((hist1, hist2, hist3))
    pred = np.squeeze(pred)
    # pred = pred.reshape((3, 256, 1))

    # img2 = cv2.imread(r"D:\TEST_IMAGE\3.jpg")
    # img2 = cv2.imread(r"D:\Datasets\yunse\train\crop_img_27_46.tif")
    img2 = cv2.imread(r"D:\Datasets\test\train\shuru\crop_6_3.jpg")
    # img2 = cv2.imread(r"D:\Datasets\test\train\moban\crop_7_3.jpg")
    hist2_1 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist2_2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
    hist2_3 = cv2.calcHist([img2], [2], None, [256], [0, 256])
    # target = np.concatenate((hist2_1, hist2_2, hist2_3))
    target = np.stack((hist2_1, hist2_2, hist2_3))
    target = np.squeeze(target)
    print(target.shape)
    print(pred.shape)
    # target = target.reshape((3, 256, 1))
    loss = L2loss(target, pred)
    print(loss)
