import argparse
import os

import sys

import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np

import models.densenet as dn
import models.wideresnet as wn
from utils import LinfPGDAttack, TinyImages
import utils.svhn_loader as svhn
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')

parser.add_argument('--quantile', default=0.125, type=float, help='quantile')

parser.add_argument('--epsilon', default=8.0, type=float, help='epsilon')
parser.add_argument('--iters', default=5, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=2.0, type=float, help='attack step size')

parser.add_argument('--pool-size', default=1000, type=int,
                    help='pool size')

parser.add_argument('--beta', default=1.0, type=float, help='beta for out_loss')

parser.add_argument('--name', required=True, type=str,
                    help='name of experiment')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--save-epoch', default=10, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--ood-batch-size', default=400, type=int,
                    help='mini-batch size (default: 400)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
np.random.seed(1)

def main():
    if args.tensorboard: configure("runs/%s"%(args.name))

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.in_dataset == "CIFAR-10":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]
        pool_size = args.pool_size
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]
        pool_size = args.pool_size
        num_classes = 100
    elif args.in_dataset == "SVHN":
        # Data loading code
        normalizer = None
        train_loader = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='train',
                                      transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='test',
                                  transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        args.epochs = 20
        args.save_epoch = 2
        lr_schedule=[10, 15, 18]
        pool_size = int(len(train_loader.dataset) * 8 / args.ood_batch_size) + 1
        num_classes = 10

    ood_dataset_size = len(train_loader.dataset) * 2

    print('OOD Dataset Size: ', ood_dataset_size)

    ood_loader = torch.utils.data.DataLoader(
        TinyImages(transform=transforms.Compose(
            [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
            batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    # create model
    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes + 1, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes + 1, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            assert False, "=> no checkpoint found at '{}'".format(args.resume)

    attack_out = LinfPGDAttack(model = model, eps=args.epsilon, nb_iter=args.iters, eps_iter=args.iter_size, targeted=False, rand_init=True, num_classes=num_classes+1, loss_func='CE', elementwise_best=True)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)

        # train for one epoch
        selected_ood_loader = select_ood(ood_loader, model, args.batch_size * 2, num_classes, pool_size, ood_dataset_size, args.quantile)

        train_atom(train_loader, selected_ood_loader, model, criterion, num_classes, optimizer, epoch, attack_out)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, num_classes)

        # remember best prec@1 and save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)

def select_ood(ood_loader, model, batch_size, num_classes, pool_size, ood_dataset_size, quantile):

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

    out_iter = iter(ood_loader)

    print('Start selecting OOD samples...')

    start = time.time()

    # select ood samples
    model.eval()
    with torch.no_grad():
        all_ood_input = []
        all_ood_conf = []
        for k in range(pool_size):

            try:
                out_set = next(out_iter)
            except StopIteration:
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0]
            output = model(input.cuda())
            conf = F.softmax(output, dim=1)[:,-1]

            all_ood_input.append(input)
            all_ood_conf.extend(conf.detach().cpu().numpy())

    all_ood_input = torch.cat(all_ood_input, 0)
    all_ood_conf = np.array(all_ood_conf)
    indices = np.argsort(all_ood_conf)

    N = all_ood_input.shape[0]
    selected_indices = indices[int(quantile*N):int(quantile*N)+ood_dataset_size]

    print('Total OOD samples: ', len(all_ood_conf))
    print('Max OOD Conf: ', np.max(all_ood_conf), 'Min OOD Conf: ', np.min(all_ood_conf), 'Average OOD Conf: ', np.mean(all_ood_conf))
    selected_ood_conf = all_ood_conf[selected_indices]
    print('Selected Max OOD Conf: ', np.max(selected_ood_conf), 'Selected Min OOD Conf: ', np.min(selected_ood_conf), 'Selected Average OOD Conf: ', np.mean(selected_ood_conf))

    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()

    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)

    print('Time: ', time.time()-start)

    return ood_train_loader


def train_atom(train_loader_in, train_loader_out, model, criterion, num_classes, optimizer, epoch, attack_out):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    out_confs = AverageMeter()
    in_confs = AverageMeter()

    in_losses = AverageMeter()
    out_losses = AverageMeter()
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()

    end = time.time()

    for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        in_len = len(in_set[0])
        out_len = len(out_set[0])

        in_input = in_set[0].cuda()
        in_target = in_set[1]
        in_target = in_target.cuda()

        out_input = out_set[0].cuda()

        out_target = out_set[1]
        out_target = out_target.cuda()

        adv_out_input = attack_out.perturb(out_input[int(out_len/2):], out_target[int(out_len/2):])

        model.train()

        cat_input = torch.cat((in_input, out_input[:int(out_len/2)], adv_out_input), 0)
        cat_output = model(cat_input)

        in_output = cat_output[:in_len]
        # in_conf = F.softmax(in_output, dim=1).max(dim=1)[0].mean()
        in_conf = F.softmax(in_output, dim=1)[:,-1].mean()
        in_confs.update(in_conf.data, in_len)
        in_loss = criterion(in_output, in_target)

        out_output = cat_output[in_len:]
        # out_conf = F.softmax(out_output, dim=1).max(dim=1)[0].mean()
        out_conf = F.softmax(out_output, dim=1)[:,-1].mean()
        out_confs.update(out_conf.data, out_len)
        out_loss = criterion(out_output, out_target)
        # out_loss = ood_criterion(out_output)

        in_losses.update(in_loss.data, in_len)
        out_losses.update(out_loss.data, out_len)

        nat_prec1 = accuracy(in_output[:,:num_classes].data, in_target, topk=(1,))[0]
        nat_top1.update(nat_prec1, in_len)

        loss = in_loss + args.beta * out_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                  'Prec@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                  'Out Loss {out_loss.val:.4f} ({out_loss.avg:.4f})\t'
                  'In Conf {in_confs.val:.4f} ({in_confs.avg:.4f})\t'
                  'OOD Conf {out_confs.val:.4f} ({out_confs.avg:.4f})'.format(
                      epoch, i, len(train_loader_in), batch_time=batch_time,
                      in_loss=in_losses, nat_top1=nat_top1,
                      out_loss=out_losses, out_confs=out_confs,
                      in_confs=in_confs))

    # log to TensorBoard
    if args.tensorboard:
        log_value('nat_train_loss', nat_losses.avg, epoch)
        log_value('nat_train_acc', nat_top1.avg, epoch)

def validate(val_loader, model, criterion, epoch, num_classes):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output[:,:num_classes].data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg

def save_checkpoint(state, epoch):
    """Saves checkpoint to disk"""
    directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)

class OODDataset(torch.utils.data.Dataset):
  def __init__(self, images, labels):
        self.labels = labels
        self.images = images

  def __len__(self):
        return len(self.images)

  def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y

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

def adjust_learning_rate(optimizer, epoch, lr_schedule):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1

    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
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
    main()
