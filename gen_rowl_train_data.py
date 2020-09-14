import sys

import numpy as np

from utils import TinyImages
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.svhn_loader as svhn
import torchvision
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

ood_loader = torch.utils.data.DataLoader(TinyImages(transform=transforms.ToTensor()), batch_size=1, shuffle=False)

save_dir = "datasets/rowl_train_data/CIFAR-10"

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i in range(11):
    os.makedirs(os.path.join(save_dir, '%02d'%i))

class_count = np.zeros(10)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                     transform=transform),
    batch_size=1, shuffle=False)

for (xs, ys) in train_loader:
    image = xs[0]
    label = ys[0].numpy()
    torchvision.utils.save_image(image, os.path.join(save_dir, '%02d'%label, '%d.png'%class_count[label]))
    class_count[label] += 1

ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

for i, (images, labels) in enumerate(ood_loader):
    torchvision.utils.save_image(images[0], os.path.join(save_dir, '10', '%d.png'%i))
    if i + 1 == 5000:
        break

save_dir = "datasets/rowl_train_data/CIFAR-100"

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i in range(101):
    os.makedirs(os.path.join(save_dir, '%03d'%i))

class_count = np.zeros(101)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                     transform=transform),
    batch_size=1, shuffle=False)

for (xs, ys) in train_loader:
    image = xs[0]
    label = ys[0].numpy()
    torchvision.utils.save_image(image, os.path.join(save_dir, '%03d'%label, '%d.png'%class_count[label]))
    class_count[label] += 1

ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

for i, (images, labels) in enumerate(ood_loader):
    torchvision.utils.save_image(images[0], os.path.join(save_dir, '100', '%d.png'%i))
    if i + 1 == 500:
        break

save_dir = "datasets/rowl_train_data/SVHN"

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i in range(11):
    os.makedirs(os.path.join(save_dir, '%02d'%i))

class_count = np.zeros(10, dtype=np.int32)

train_loader = torch.utils.data.DataLoader(
    svhn.SVHN('datasets/svhn/', split='train', transform=transform, download=False),
    batch_size=1, shuffle=True)

for (xs, ys) in train_loader:
    image = xs[0]
    label = ys[0].numpy()
    torchvision.utils.save_image(image, os.path.join(save_dir, '%02d'%label, '%d.png'%class_count[label]))
    class_count[label] += 1

ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

print('SVHN, Class count: ', class_count)
ood_count = int(np.mean(class_count))
print('SVHN, OOD count: ', ood_count)

for i, (images, labels) in enumerate(ood_loader):
    torchvision.utils.save_image(images[0], os.path.join(save_dir, '10', '%d.png'%i))
    if i + 1 == ood_count:
        break