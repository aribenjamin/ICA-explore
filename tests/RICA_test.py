
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

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Rica_Net(nn.Module):

    def __init__(self):
        super(Rica_Net, self).__init__()
        self.linear_ica = ICALinear(32*32, 32, ica_strength = 1e-1, super_or_sub = "both")


    def forward(self, x):

        output = self.linear_ica(x)

        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=256)
    parser.add_argument('--nEpochs', type=int, default=20)
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
        transforms.CenterCrop(32),
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

    optimizer = optim.Adam(net.parameters(), lr=1e-3,
                              weight_decay=0)


    for epoch in range(1, args.nEpochs + 1):
        train(args, epoch, net, trainLoader, optimizer)


    analyze_filters(args, net, testLoader)
    torch.save(net.state_dict(), "saved_RICA_net.pt")

def analyze_filters(args, net, testLoader):
    """The weights of this net are a 32^2 x 32^2 matrix. We'll visualize a handful.
    We'll also examine the distributions of these filters"""
    histograms = get_test_dists(args, net, testLoader)
    for idx in range(3):
        plt.figure(figsize=(10,5))
        filt = net.linear_ica.weight[idx,:].view(32,32)
        plt.subplot(121)
        plt.imshow(filt.detach().cpu().numpy())

        plt.subplot(122)
        x = histograms[idx].detach().cpu().numpy()
        plt.semilogy(x)
        xmin = np.where(x>0)[0][0]
        xmax = np.where(x>0)[0][-1]
        plt.xlim([xmin,xmax])
        plt.savefig("filter_{}.png".format(idx))
        plt.show()



def train(args, epoch, net, trainLoader, optimizer):
    net.train()

    for batch_idx, (data, _) in enumerate(trainLoader):

        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        data = torch.mean(data, 1).view(-1,32**2)

        optimizer.zero_grad()
        output = torch.squeeze(net(data))
        # now reconstruct the input
        data_r = output.mm(net.linear_ica.weight)

        # let's just put the loss right in here
        mse_loss = F.mse_loss(data,data_r)
        mse_loss.backward()
        
        nongaussianity = torch.mean(torch.log(torch.cosh(output)))
        
        ## Optional check: same as if you added the ICA term as a cost?
        # see that they're similar
        g1 = net.linear_ica.weight.grad.detach().clone().cpu().numpy()
        print(g1)
        optimizer.zero_grad()
        
        # do a forward pass by hand
        output = data.mm(net.linear_ica.weight.t())
        loss2 = torch.mean(torch.log(torch.cosh(output)))
        loss2.backward()
        # see that they're similar
        g2 = net.linear_ica.weight.grad.detach().clone().cpu().numpy()
        print("RICA",g2)
        print("ratio",g1/g2)
        
        optimizer.step()

        print('Train Epoch {}: Loss: {:.6f},\t Nongaussianity: {:.6f}\t'.format(epoch,
            mse_loss.item(), nongaussianity.item()))



def get_test_dists(args, net, testLoader, n_bins = 1000, maxval = 50):
    net.eval()
    histograms = [torch.zeros(n_bins) for channel in range(32**2)]
    for data, target in testLoader:
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
            data = torch.mean(data, 1).view(-1,32**2)
            output = torch.squeeze(net(data))

        cpu_output = output.cpu()
        for channel in range(32):
            binned = torch.histc(cpu_output[:, channel], bins=n_bins, min=-maxval, max=maxval)
            histograms[channel] = histograms[channel] + torch.squeeze(binned)
    return histograms

if __name__ == '__main__':
    main()
