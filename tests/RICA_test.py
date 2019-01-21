
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
from ICA_linear import ICALinear


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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Rica_Net(nn.Module):

    def __init__(self):
        super(Rica_Net, self).__init__()
        self.linear_ica = ICALinear(32*32, 32*32)


    def forward(self, x):

        output = self.linear_ica(x)

        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=256)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop', 'sgdw'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'error.csv'

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

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

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        dset.CIFAR10(root='./data', train=True, download=True,
                     transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)

    testLoader = DataLoader(
        dset.CIFAR10(root='./data', train=False, download=True,
                     transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    net = Rica_Net()

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=1e-1,
                              momentum=0.9, weight_decay=1e-4)


    for epoch in range(1, args.nEpochs + 1):
        train(args, epoch, net, trainLoader, optimizer)


    analyze_filters(args, net, testLoader)

def analyze_filters(args, net, testLoader):
    """The weights of this net are a 32^2 x 32^2 matrix. We'll visualize a handful.
    We'll also examine the distributions of these filters"""
    histograms = get_test_dists(args, net, testLoader)
    for idx in range(10):
        plt.figure(figsize=(10,5))
        filt = net.weight[idx,:].view(32,32)
        plt.subplot(121)
        plt.imshow(filt)

        plt.subplot(122)
        plt.plot(histograms[idx])
        plt.savefig("filter_{}.png".format(idx))
        plt.show()



def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()

    for batch_idx, (data, _) in enumerate(trainLoader):

        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        data = torch.mean(data, 1).flatten()

        optimizer.zero_grad()
        output = torch.squeeze(net(data))
        # now reconstruct the input
        data_r = net.linear_ica.weight.t().mm(output)

        # let's just put the loss right in here
        loss = F.mse_loss(data,data_r)

        loss.backward()

        optimizer.step()


        print('Train Epoch: Loss: {:.6f}\t'.format(
            loss.item()))



def get_test_dists(args, net, testLoader, n_bins = 1000, maxval = 50):
    net.eval()
    histograms = [torch.zeros(n_bins) for channel in range(32**2)]
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = torch.squeeze(net(data))

        cpu_output = output.cpu()
        for channel in range(32**2):
            binned = torch.histc(cpu_output[:, channel], bins=n_bins, min=-maxval, max=maxval)
            histograms[channel] = histograms[channel] + torch.squeeze(binned)
    return histograms

if __name__ == '__main__':
    main()
