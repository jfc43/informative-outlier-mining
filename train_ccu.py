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
import models.gmm as gmmlib

from utils import TinyImages
import utils.svhn_loader as svhn
from sklearn import mixture

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--save-epoch', default=10, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--ood-batch-size', default=50, type=int,
                    help='mini-batch size (default: 50)')
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

parser.add_argument('--name', required=True, type=str,
                    help='name of experiment')
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
        num_classes = 10

    out_loader = torch.utils.data.DataLoader(
    TinyImages(transform=transforms.Compose(
        [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
        batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    # create model
    if args.model_arch == 'densenet':
        base_model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        base_model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    gen_gmm(train_loader, out_loader, data_used=50000, PCA=True, N=[100])

    gmm = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'in_gmm.pth.tar')

    gmm.alpha = nn.Parameter(gmm.alpha)
    gmm.mu.requires_grad = True
    gmm.logvar.requires_grad = True
    gmm.alpha.requires_grad = False

    gmm_out = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'out_gmm.pth.tar')
    gmm_out.alpha = nn.Parameter(gmm.alpha)
    gmm_out.mu.requires_grad = True
    gmm_out.logvar.requires_grad = True
    gmm_out.alpha.requires_grad = False
    loglam = 0.
    model = gmmlib.DoublyRobustModel(base_model, gmm, gmm_out,
                                     loglam, dim=3072,
                                     classes=num_classes).cuda()

    model.loglam.requires_grad = False

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

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
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    lr = args.lr
    lr_gmm = 1e-5

    param_groups = [{'params':model.mm.parameters(),'lr':lr_gmm, 'weight_decay':0.},
                    {'params':model.mm_out.parameters(),'lr':lr_gmm, 'weight_decay':0.},
                    {'params':model.base_model.parameters(),'lr':lr, 'weight_decay':args.weight_decay}]


    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, nesterov=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)

        # train for one epoch
        lam = model.loglam.data.exp().item()
        train_CEDA_gmm_out(model, train_loader, out_loader, optimizer, epoch, lam=lam)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)


def gen_gmm(train_loader, out_loader, data_used=50000, PCA=True, N=[100]):

    print('Generate GMM...')
    start = time.time()

    dim = 3072

    X = []
    for x, f in train_loader:
        X.append(x.view(-1,dim))
    X = torch.cat(X, 0)

    X = X[:data_used] #needed to keep memory of distance matrix below 800 GB

    if PCA:
        metric = gmmlib.PCAMetric(X, p=2, min_sv_factor=1e6)
        X = ( (X@metric.comp_vecs.t()) / metric.singular_values_sqrt[None,:] )
    else:
        metric = gmmlib.LpMetric()

    for n in N:
        print(n)
        gmm = gmmlib.GMM(n, dim, metric=metric)

        clf = mixture.GMM(n_components=n, covariance_type='spherical', params='mc')

        clf.fit(X)
        mu = torch.tensor(clf.means_ ,dtype=torch.float)

        logvar = torch.tensor(np.log(clf.covars_[:,0]) ,dtype=torch.float)
        logvar = 0.*logvar + logvar.exp().mean().log()

        alpha = torch.tensor(np.log(clf.weights_) ,dtype=torch.float)
        gmm = gmmlib.GMM(n, dim, mu=mu, logvar=logvar, metric=metric)


        if PCA:
            gmm.mu.data = ( (gmm.mu.data * metric.singular_values_sqrt[None,:] )
                           @ metric.comp_vecs.t().inverse() )

        directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + 'in_gmm.pth.tar'
        torch.save(gmm, filename)

    X = []
    for idx, (x, f) in enumerate(out_loader):
        if idx>400:
            break;
        X.append(x.view(-1,dim))
    X = torch.cat(X, 0)

    if PCA:
        X = ( (X@metric.comp_vecs.t()) / metric.singular_values_sqrt[None,:] )

    for n in N:
        print(n)
        # Out GMM
        gmm = gmmlib.GMM(n, dim, metric=metric)

        clf = mixture.GMM(n_components=n, covariance_type='spherical', params='mc')

        clf.fit(X)
        mu = torch.tensor(clf.means_ ,dtype=torch.float)

        logvar = torch.tensor(np.log(clf.covars_[:,0]) ,dtype=torch.float)
        logvar = 0.*logvar + logvar.exp().mean().log()

        alpha = torch.tensor(np.log(clf.weights_) ,dtype=torch.float)
        gmm = gmmlib.GMM(n, dim, mu=mu, logvar=logvar, metric=metric)

        if PCA:
            gmm.mu.data = ( (gmm.mu.data * metric.singular_values_sqrt[None,:] )
                           @ metric.comp_vecs.t().inverse() )

        directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + 'out_gmm.pth.tar'
        torch.save(gmm, filename)

    print('Time: ', time.time() - start)

    print('Done!')

def train_CEDA_gmm_out(model, train_loader, ood_loader, optimizer, epoch, lam=1., verbose=10):
    criterion = nn.NLLLoss()

    model.train()

    train_loss = 0
    likelihood_loss = 0
    correct = 0
    margin = np.log(4.)

    if ood_loader is not None:
        ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))
        ood_loader_iter = iter(ood_loader)

    p_in = torch.tensor(1. / (1. + lam), dtype=torch.float).cuda()
    p_out = torch.tensor(lam, dtype=torch.float).cuda() * p_in

    log_p_in = p_in.log()
    log_p_out = p_out.log()

    start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        noise = next(ood_loader_iter)[0].cuda()

        optimizer.zero_grad()

        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        full_out = F.log_softmax(full_out, dim=1)

        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]:]

        like_in_in = torch.logsumexp(model.mm(data.view(data.shape[0], -1)), 0 )
        like_out_in =  torch.logsumexp(model.mm(noise.view(noise.shape[0], -1)), 0 )

        like_in_out = torch.logsumexp(model.mm_out(data.view(data.shape[0], -1)), 0 )
        like_out_out =  torch.logsumexp(model.mm_out(noise.view(noise.shape[0], -1)), 0 )

        loss1 = criterion(output, target)
        loss2 = -output_adv.mean()
        loss3 = - torch.logsumexp(torch.stack([log_p_in + like_in_in,
                                               log_p_out + like_in_out], 0), 0).mean()
        loss4 = - torch.logsumexp(torch.stack([log_p_in + like_out_in,
                                               log_p_out + like_out_out], 0), 0).mean()

        loss =  p_in*(loss1 + loss3) + p_out*(loss2 + loss4)

        loss.backward()
        optimizer.step()

        likelihood_loss += loss3.item()
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        threshold = model.mm.logvar.max() + margin
        idx = model.mm_out.logvar<threshold
        model.mm_out.logvar.data[idx] = threshold

        if (batch_idx % verbose == 0) and verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('Time: ', time.time() - start)

def validate(val_loader, model, criterion, epoch):
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
        prec1 = accuracy(output.data, target, topk=(1,))[0]
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

def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""

    if epoch in lr_schedule:
        for group in optimizer.param_groups:
            group['lr'] *= .1

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
