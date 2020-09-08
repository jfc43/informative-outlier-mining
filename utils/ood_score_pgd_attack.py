
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
from scipy import misc

class OODScoreLoss(nn.Module):
    def __init__(self, num_classes):
        super(OODScoreLoss, self).__init__()
        self.num_classes=num_classes

    def forward(self, x, y=None):
        return nn.CrossEntropyLoss(reduction='none')(x, (torch.ones(x.shape[0], dtype=torch.long)*self.num_classes).cuda())

class OODScoreLinfPGDAttack:
    def __init__(
            self, model, eps=4.0, nb_iter=40,
            eps_iter=1.0, rand_init=True, clip_min=0., clip_max=1.,
            num_classes = 10, n_restarts=1):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.model = model

        self.detector_loss_func = OODScoreLoss(num_classes=num_classes)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.num_classes = num_classes
        self.n_restarts = n_restarts

    def get_loss(self, x):
        outputs = self.model(x)
        loss = self.detector_loss_func(outputs)

        return loss

    def attack_single_run(self, x):

        x = x.detach().clone()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        with torch.no_grad():
            loss = self.get_loss(x)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-1, 1)
            delta.data *= self.eps
            delta.data = torch.round(delta.data)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

        for ii in range(self.nb_iter):
            adv_x = x + delta / 255.0
            loss = self.get_loss(adv_x)

            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

            delta.grad.data.zero_()

        with torch.no_grad():
            adv_x = x + delta / 255.0
            loss = self.get_loss(adv_x)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]

        return worst_perb, worst_loss
    
    def perturb(self, x):
        """
        Given examples x, returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        with torch.no_grad():
            loss = self.get_loss(x)
            worst_loss = loss.data.clone()
            worst_perb = torch.zeros_like(x)

        for k in range(self.n_restarts):
            delta, loss = self.attack_single_run(x)
            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]

        adv_x = torch.clamp(x + torch.clamp(torch.round(worst_perb), min=-self.eps, max=self.eps) / 255.0, min=self.clip_min, max=self.clip_max)

        return adv_x
