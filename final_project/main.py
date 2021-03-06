import torch
import time
from datetime import datetime
import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import math
import pickle
from final_project import vgg
from tqdm import tqdm
from final_project.preprocessing import pre_process_dataset

# insert your desired saving path here:
save_dir = r"C:\Users\omri_\Downloads\train_videos\saving_dir/"
# insert path with train videos, or train data pre-processed pickle, here:
train_input_path = r"C:\Users\omri_\Downloads\train_videos/"
# insert path with test videos, or test data pre-processed pickle, here:
test_input_path = r"C:\Users\omri_\Downloads\test_videos/"

model_names = 'vgg19'
batch_size = 128
workers = 4
momentum = 0.9
lr = 0.01
weight_decay = 5e-4
start_epoch = 0
epochs = 20
print_freq = 2
vid_acc_list = []
res_list = []
res_list_av = []


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

    for i, (input, target) in tqdm(enumerate(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

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

    return (top1.avg, top1.val)


def save_checkpoint(input_path, data_dict):
    """
    Save the training model
    """
    time_str = data_dict.get("time_of_epoch")
    time_str = re.sub("[ \-:]","_",time_str.__str__()[:-7])
    with open(input_path+time_str+"_epoch"+str(data_dict.get("epoch"))+"_data.pkl", "wb") as file:
        pickle.dump(data_dict, file)

    res_list.append(data_dict.get("prec1"))
    res_list_av.append(data_dict.get("prec1_av"))


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
    lr = lr * (0.5 ** (epoch // 10))
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


def our_loader(dataset, dataset_labels):
    image_torch_list = []
    label_torch_list = []
    for i in range(math.ceil(dataset.shape[0]/batch_size)):
        image_torch_list.append(dataset[(128*i):((128*(i+1)))])
        label_torch_list.append(dataset_labels[(128*i):((128*(i+1)))])

    return list(zip(image_torch_list, label_torch_list))


def video_preds_acc(val_loader, val_video_labels, pred_model):

    all_outputs = []
    all_labels = []
    for i, (input, target) in tqdm(enumerate(val_loader)):
        output = pred_model(input)
        output = output.detach().numpy()
        output = pd.DataFrame(output).apply(lambda x: math.exp(x.loc[0])/(math.exp(x.loc[0]) + math.exp(x.loc[1])), 1)
        target = pd.Series(target.numpy())
        all_outputs.extend(output.tolist())
        all_labels.extend(target.tolist())

    df = pd.DataFrame(zip(all_outputs, all_labels, val_video_labels), columns=["proba", "label", "video_name"])
    video_means = df.groupby(["video_name"]).mean()
    thr = sorted(video_means.proba)[round(video_means.shape[0]*0.164327)-1]
    video_means["final_pred"] = video_means["proba"] <= thr

    final_acc = (video_means.final_pred == video_means.label).mean()
    return final_acc


def main():
    best_prec1 = 0

    # Check the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = vgg.__dict__['vgg19']()

    train_dataset, train_dataset_labels, train_video_labels = pre_process_dataset(train_input_path, "train")
    test_dataset, test_dataset_labels, test_video_labels = pre_process_dataset(test_input_path, "test")

    train_loader = our_loader(train_dataset, train_dataset_labels)
    test_loader = our_loader(test_dataset, test_dataset_labels)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1/0.835673, 1/0.164327]))
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    print("Start Training")
    for epoch in tqdm(range(start_epoch, epochs)):

        # adjusting L.R
        adjust_learning_rate(optimizer, epoch, lr)

        # training model
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec1_current = validate(test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        best_prec1 = max(prec1, best_prec1)

        # video predictions accuracy
        vid_acc = video_preds_acc(test_loader,test_video_labels, model)
        vid_acc_list.append(vid_acc)
        print("Current video accuracy: "+str(vid_acc))

        save_checkpoint(save_dir,
            {'time_of_epoch': datetime.now(),
             'epoch': epoch,
             'state_dict': model.state_dict(),
             'best_prec1': best_prec1,
             'prec1_av': prec1,
             'prec1':prec1_current,
             'video_acc': vid_acc})

        df = pd.DataFrame(zip(range(len(res_list)), res_list), columns=["epoch", "acc"])
        df.to_csv(save_dir+"accuracy_per_epoch_on_val_"+str(epoch)+".csv", index=None)

        df_av = pd.DataFrame(zip(range(len(res_list_av)),res_list_av), columns=["epoch", "acc"])
        df_av.to_csv(save_dir+"accuracy_per_epoch_average_on_val_"+str(epoch)+".csv", index=None)

        df_vid_acc = pd.DataFrame(zip(range(len(vid_acc_list)),vid_acc_list), columns=["epoch", "acc"])
        df_vid_acc.to_csv(save_dir+"video_accuracy"+str(epoch)+".csv", index=None)

        fig = df.plot(kind='line', x='epoch', y='acc').get_figure()
        fig.savefig(save_dir+"accuracy_per_epoch_on_val_plt_"+str(epoch)+".jpg")

        fig = df_av.plot(kind='line', x='epoch', y='acc').get_figure()
        fig.savefig(save_dir+"accuracy_per_epoch_on_val_av_plt_"+str(epoch)+".jpg")

        fig = df_vid_acc.plot(kind='line', x='epoch', y='acc').get_figure()
        fig.savefig(save_dir+"video_accuracy_per_epoch_plt_"+str(epoch)+".jpg")


if __name__ == '__main__':
    main()
