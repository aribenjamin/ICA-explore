#!/usr/bin/env python
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.autograd import Variable
import torchvision.models as models
import shutil

import setproctitle

from torch.optim import Optimizer
required = object()


import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import os

from orion.client import report_results

class Net(nn.Module):
    def __init__(self, ica_strengths = [1e-1]*4 + [0]*2):
        super(Net, self).__init__()

        self.ica_strengths = ica_strengths
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        nongaussianities = 0

        x = self.conv1(x)
        nongaussianities += torch.mean(torch.log(torch.cosh(x)))*self.ica_strengths[0]
        x = self.conv2(F.relu(x))
        nongaussianities += torch.mean(torch.log(torch.cosh(x)))*self.ica_strengths[1]
        x = F.max_pool2d(F.relu(x), 2)


        x = self.conv3(x)
        nongaussianities += torch.mean(torch.log(torch.cosh(x)))*self.ica_strengths[2]
        x = self.conv4(F.relu(x))
        nongaussianities += torch.mean(torch.log(torch.cosh(x)))*self.ica_strengths[3]
        x = F.max_pool2d(F.relu(x), 2)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        nongaussianities += torch.mean(torch.log(torch.cosh(x)))*self.ica_strengths[4]

        x = self.fc2(F.relu(x))
        nongaussianities += torch.mean(torch.log(torch.cosh(x)))*self.ica_strengths[5]

        

        return F.log_softmax(x), nongaussianities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=256)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--card', type=int, default=2)

    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--ica', type=float, default=1e-1)
    parser.add_argument('--ica-fc', type=float, default=0)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop', 'sgdw'))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.card)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'error.csv'


    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        dset.CIFAR10(root='./data', train=True, download=True,
                     transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)

    testLoader = DataLoader(
        dset.CIFAR10(root='./data', train=False, download=True,
                     transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    net = Net([args.ica]*4 + [args.ica_fc]*2)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()
        #net = nn.DataParallel(net, device_ids=[0,1])

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=args.wd)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)
        

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test_error_rate = test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))

    trainF.close()
    testF.close()

    report_results([dict(
        name='test_error_rate',
        type='objective',
        value=test_error_rate)])

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
    
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        
        optimizer.zero_grad()
        output, nongaussianity = net(data)
        loss = F.nll_loss(torch.squeeze(output), target) + nongaussianity
        loss.backward()
        
        optimizer.step()
        
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/float(len(data))
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output, _ = net(data)
        test_loss += F.nll_loss(torch.squeeze(output), target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= float(len(testLoader)) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/float(nTotal)
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

    return float(incorrect)/float(nTotal)

def adjust_opt(optAlg, optimizer, epoch):
    if epoch == 150 or epoch == 225: 
        for param_group in optimizer.param_groups:
            if optAlg == 'sgdw':
                param_group['inner_lr'] /= 10.
            param_group['lr'] /= 10.
        



if __name__=='__main__':
    main()
