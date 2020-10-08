import argparse
import os
import shutil
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg
from tqdm import tqdm
import os
import re
import pandas as pd
import numpy as np

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

"""
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
"""

save_dir = r"C:\Users\Bengal\Desktop\project/vgg/"
model_names = 'vgg19'
batch_size = 128
workers = 4
best_prec1 = 0
momentum = 0.9
lr = 0.05
weight_decay = 5e-4
start_epoch = 0
epochs = 1  # 300
print_freq = 20

def main(evaluate):
    save_dir = r"C:\Users\Bengal\Desktop\project/vgg/"
    model_names = 'vgg19'
    batch_size = 128
    workers = 4
    best_prec1 = 0
    momentum = 0.9
    lr = 0.05
    weight_decay = 5e-4
    start_epoch = 0
    epochs = 1  # 300
    print_freq = 20

    # global args, best_prec1
    # args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = vgg.__dict__['vgg19']()

    model.features = torch.nn.DataParallel(model.features)

    # optionally resume from a checkpoint

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    ####### my addings
    # upload our data
    print("Uploading Data")
    input_path = r"C:\Users\Bengal\Desktop\project/"
    train_sample_metadata = pd.read_json(input_path + 'metadata.json').T
    resized_output_path = input_path + "all_train_imgs/"

    train_images = os.listdir(resized_output_path)
    train_labels = [train_sample_metadata.loc[vid + ".mp4"]['label'] for vid in
                    [re.sub('[0-9]+.jpg', '', item) for item in os.listdir(resized_output_path)]]
    train_labels_df = pd.DataFrame(zip(train_images, train_labels), columns=["image", "label"])

    labels = (train_labels_df["label"] == 'REAL').apply(int)

    tr_im = []
    i=0
    for img in tqdm(resized_output_path + train_labels_df["image"]):
        img = cv2.imread(img)
        img = np.reshape(img,(256,256,3))
        tr_im.append([img])#,labels[i]])
        i += 1


    tr_im = np.squeeze(np.array(tr_im), axis=(1,))
    print('X shape: ',tr_im.shape)
    print("Tensor the data")
    ###
    from torch.utils.data import TensorDataset
    tr_im_t = torch.Tensor(tr_im)
    lab = torch.Tensor(labels)
    my_dataset = TensorDataset(tr_im_t, lab)  # create your datset
    ###

    # tr=np.reshape(tr_im,(7442,256,256,3))
    ###
    # train_data = []
    # for i in range(len(tr_im)):
    #     train_data.append([tr_im[i], labels[i]])

    # trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
    #                                           num_workers=workers, pin_memory=True)
    # i1, l1 = next(iter(trainloader))
    # print(i1.shape)
    ######
    #
    # from torch.utils.data import TensorDataset
    #
    # my_x = tr_im  # a list of numpy arrays
    # my_y = labels  # another list of numpy arrays (targets)
    #
    # tensor_x = torch.Tensor(my_x)  # transform to torch tensor
    # tensor_y = torch.Tensor(my_y)
    #
    # my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # #my_dataloader = DataLoader(my_dataset)  # create your dataloader
    print("DataLoader")
    train_loader = torch.utils.data.DataLoader(
        #tr_im,
        my_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    # untill here

    # cifar = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, 4),
    #     transforms.ToTensor(),
    #     normalize,
    # ]), download=True)

    """
    # data_path = input_path + "all_train_imgs/"
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    """
    #######
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    """
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    # if args.cpu:
    #     criterion = criterion.cpu()
    # else:
    #     criterion = criterion.cuda()
    #
    # if args.half:
    #     model.half()
    #     criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    if evaluate:
        validate(val_loader, model, criterion)
        return
    print("Start Training")
    for epoch in tqdm(range(start_epoch, epochs)):

        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch

        # for i in range(batch_size):
        #      # Local batches and labels
        #     local_X, local_y = tr_im[i * batch_size:(i + 1) * batch_size, ], y[i * batch_size:(i + 1) * batch_size, ]

        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(save_dir, 'checkpoint_{}.tar'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()


    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # if args.cpu == False:
        #     input = input.cuda(async=True)
        #     target = target.cuda(async=True)
        # if args.half:
        #     input = input.half()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # if args.cpu == False:
        #     input = input.cuda(async=True)
        #     target = target.cuda(async=True)
        #
        # if args.half:
        #     input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main(evaluate = False)
