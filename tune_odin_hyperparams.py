from __future__ import print_function
import argparse
import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import models.densenet as dn
import models.wideresnet as wn
import utils.svhn_loader as svhn
import numpy as np
import time
from scipy import misc
from utils import ConfidenceLinfPGDAttack, MahalanobisLinfPGDAttack, softmax, metric, sample_estimator, get_Mahalanobis_score, TinyImages, CustomDataset

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--name', required=True, type=str,
                    help='neural network name and training set')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')

parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')

parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')

parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')


def get_odin_score(inputs, model, temper, noiseMagnitude1):

    criterion = nn.CrossEntropyLoss()
    inputs = Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores

def tune_odin_hyperparams():
    print('Tuning hyper-parameters...')
    stypes = ['ODIN']

    save_dir = os.path.join('output/odin_hyperparams/', args.in_dataset, args.name, 'tmp')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        trainset= torchvision.datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        trainset= torchvision.datasets.CIFAR100('./datasets/cifar100', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

        num_classes = 100

    elif args.in_dataset == "SVHN":

        normalizer = None
        trainloaderIn = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='train',
                                      transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=True)
        testloaderIn = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='test',
                                  transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=True)

        args.epochs = 20
        num_classes = 10

    valloaderOut = torch.utils.data.DataLoader(TinyImages(transform=transforms.Compose(
        [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=False)

    valloaderOut.dataset.offset = np.random.randint(len(valloaderOut.dataset))

    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs=args.epochs))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.cuda()

    m = 1000
    val_in = []
    val_out = []

    cnt = 0
    for data, target in testloaderIn:
        for x in data:
            val_in.append(x.numpy())
            cnt += 1
            if cnt == m:
                break
        if cnt == m:
            break

    cnt = 0
    for data, target in valloaderOut:
        for x in data:
            val_out.append(x.numpy())
            cnt += 1
            if cnt == m:
                break
        if cnt == m:
            break

    print('Len of val in: ', len(val_in))
    print('Len of val out: ', len(val_out))

    best_fpr = 1.1
    best_magnitude = 0.0
    for magnitude in np.arange(0, 0.0041, 0.004/20):

        t0 = time.time()
        f1 = open(os.path.join(save_dir, "confidence_ODIN_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, "confidence_ODIN_Out.txt"), 'w')
    ########################################In-distribution###########################################
        print("Processing in-distribution images")

        count = 0
        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)])
            images = images.cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            scores = get_odin_score(images, model, temper=1000, noiseMagnitude1=magnitude)

            for k in range(batch_size):
                f1.write("{}\n".format(scores[k]))

            count += batch_size
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)])
            images = images.cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            scores = get_odin_score(images, model, temper=1000, noiseMagnitude1=magnitude)

            for k in range(batch_size):
                f2.write("{}\n".format(scores[k]))

            count += batch_size
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        f1.close()
        f2.close()

        results = metric(save_dir, stypes)
        print_results(results, stypes)
        fpr = results['ODIN']['FPR']
        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude

    return best_magnitude

if __name__ == '__main__':
    best_magnitude = tune_odin_hyperparams()
    print('Best magnitude: ', best_magnitude)