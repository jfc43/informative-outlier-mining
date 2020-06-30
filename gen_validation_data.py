import sys

import numpy as np

from utils import TinyImages
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

ood_loader = torch.utils.data.DataLoader(TinyImages(transform=transforms.ToTensor()), batch_size=1, shuffle=False)

ood_loader.dataset.offset = 0

save_dir = "datasets/val_ood_data/tiny_images/0"

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i, (images, labels) in enumerate(ood_loader):
    torchvision.utils.save_image(images[0], os.path.join(save_dir, '%d.png'%i))
    if i + 1 == 10000:
        break
