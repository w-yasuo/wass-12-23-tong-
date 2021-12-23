import os
import PIL
import time
import torch
import argparse
import numpy as np

from torch import nn
from PIL import Image
from visdom import Visdom
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import L2loss
from model import SqueezeNet
from data import SelfDatasetFolder
from utils import adjust_learning_rate, accuracy, save_checkpoint, AverageMeter, ProgressMeter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR', default=r'D:\Datasets\liaoning',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='SqueezeNet',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=900, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# best_acc1 = 0
best_loss = 500


# wind = Visdom()
# wind.line([[0., 0.]], [0.], win='train', opts=dict(title='loss', legend=['loss']))


def main():
    global best_loss
    args = parser.parse_args()
    # 加载模型
    model = SqueezeNet(args)
    # print(model)

    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Data loader
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    train_dataset = SelfDatasetFolder(
        traindir, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_transforms = transforms.Compose([
        transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize, ])
    print('Using image size', args.image_size)

    val_loader = torch.utils.data.DataLoader(
        SelfDatasetFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, L2loss, optimizer, epoch, args)

        # evaluate on validation set
        loss_1 = validate(val_loader, model, L2loss, args)

        # remember best acc@1 and save checkpoint
        is_best = loss_1 < best_loss
        best_loss = min(loss_1, best_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % 1 == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, L2loss, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, path, target) in enumerate(train_loader):

        # print("*******T*********")
        # print(type(target))  # <class 'torch.Tensor'>
        # print(target.shape)  # torch.Size([48, 3, 256])
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)  # torch.Size([3, 256, 2])
        # output = output.permute(1, 2, 0)
        # print("*******P*********")
        # print(type(output))  # <class 'torch.Tensor'>
        # print(output.shape)  # torch.Size([48, 3, 256])

        # measure accuracy and record loss

        # print(target, output)
        # quit()
        loss = L2loss(target, output)
        loss = np.array(loss, dtype=np.float32)
        loss = torch.from_numpy(loss).requires_grad_()
        # wind.line([[loss]], [i], win='train', update='append')
        losses.update(loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, L2loss, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(val_loader), batch_time, losses, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, path, target) in enumerate(val_loader):
            # bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)
            # target = bn(target)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            # output = torch.unsqueeze(output, dim=2)  # torch.Size([3, 256, 1, 2])
            # output = output.permute(3, 0, 1, 2)

            loss = L2loss(target, output)
            loss = np.array(loss, dtype=np.float32)
            loss = torch.from_numpy(loss).requires_grad_()
            losses.update(loss)
            # measure accuracy and record loss

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        print(' ** loss: {} ** '.format(losses))
        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

    return loss


if __name__ == '__main__':
    # image = Image.open(r"D:\TEST_IMAGE\2.jpg")
    # image = image.resize((224, 224))
    # toTensor = transforms.ToTensor()  # 实例化一个toTensor
    # image_tensor = toTensor(image)
    # image_tensor = image_tensor.reshape(1, 3, 224, 224)
    # # print("model:", model)
    # output1, output2, output3 = model(image_tensor)
    # print("output1:", output1)
    # print("output2:", output2)
    # print("output3:", output3)
    main()
