'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from datetime import datetime

from models import *
from utils import progress_bar
from PIL import Image
import ssl
import matplotlib.pyplot as plt
from dataloader_c10h import get_loader

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--cuda', default=1, type=int, help='device number')
parser.add_argument('--type', default='full', type=str)
parser.add_argument('--epoch', default=150, type=int)
parser.add_argument('--model', default='vgg', type=str)
args = parser.parse_args()
cuda_idx = args.cuda

device = torch.device('cuda:'+str(cuda_idx) if torch.cuda.is_available() else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def cross_entropy_loss(inputs, target, size_average=True):
    inputs = F.log_softmax(inputs, dim=1)
    loss = -torch.sum(inputs * target)
    if size_average:
        return loss / inputs.size(0)
    else:
        return loss


class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, inputs, target):
        return cross_entropy_loss(inputs, target, self.size_average)


# Training
def train_soft(net, optim, crit, train_loader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, targets_hard) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optim.zero_grad()
        outputs = net(inputs)
        loss = crit(outputs, targets)
        loss.backward()
        optim.step()
        # scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, answer = targets.max(1)
        total += targets.size(0)
        correct += answer.eq(predicted).sum().item()

    print(f"TRAIN| Loss: {train_loss/len(train_loader):.3f} | Acc: {100.*correct/total:.2f} %")
    return net, train_loss/len(train_loader), 100.*correct/total


def train_hard(net, optim, crit, train_loader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, _, targets_hard) in enumerate(train_loader):
        inputs, targets_hard = inputs.to(device), targets_hard.to(device)

        optim.zero_grad()
        outputs = net(inputs)
        loss = crit(outputs, targets_hard)
        loss.backward()
        optim.step()
        # scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets_hard.size(0)
        correct += targets_hard.eq(predicted).sum().item()

    print(f"TRAIN| Loss: {train_loss/len(train_loader):.3f} | Acc: {100.*correct/total:.2f} %")
    return net, train_loss/len(train_loader), 100.*correct/total


def val(net, crit, val_loader):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets_hard) in enumerate(val_loader):
            inputs, targets_hard = inputs.to(device), targets_hard.to(device)
            outputs = net(inputs)
            loss = crit(outputs, targets_hard)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets_hard.size(0)
            correct += targets_hard.eq(predicted).sum().item()

    print(f"VAL  | Loss: {val_loss/len(val_loader):.3f} | Acc: {100.*correct/total:.2f} %")


def test(net, crit, test_loader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = crit(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += targets.eq(predicted).sum().item()

    print(f"TEST | Loss: {test_loss / len(test_loader):.3f} | Acc: {100. * correct / total:.2f} %")


if __name__ == "__main__":

    if args.type == 'top2':
        train_loader, val_loader, test_loader = get_loader(top2=True)
    else:
        train_loader, val_loader, test_loader = get_loader(top2=False)

    epoch = args.epoch
    if args.model == 'vgg':
        net = VGG('VGG19')
    elif args.model == 'resnet':
        net = ResNet18()

    net = net.to(device)

    if args.type == 'hard':
        train_criterion = nn.CrossEntropyLoss()
    else:
        train_criterion = CrossEntropyLoss()

    test_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    if args.type == 'hard':
        for epoch in range(args.epoch):
            print('\nEpoch: %d' % epoch)
            net, train_loss, train_acc = train_hard(net, optimizer, train_criterion, train_loader)
            val(net, test_criterion, val_loader)
    else:
        for epoch in range(args.epoch):
            print('\nEpoch: %d' % epoch)
            net, train_loss, train_acc = train_soft(net, optimizer, train_criterion, train_loader)
            val(net, test_criterion, val_loader)

    test(net, test_criterion, test_loader)