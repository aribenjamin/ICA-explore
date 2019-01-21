import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ICA_linear import ICALinear

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 100

# Create random Tensors to hold input and outputs.
## supergaussian inputs
x = torch.exp(torch.randn(N, D_in, device=device, dtype=dtype))
w = torch.randn(D_in, D_out, device=device, dtype=dtype, requires_grad=False)
NOISE = 0.01
y = torch.matmul(x, w) + torch.randn(N, D_out, device=device, dtype=dtype)*NOISE

# Create random Tensors for weights.
w1 = torch.randn(D_in, D_out, device=device, dtype=dtype, requires_grad=True)

# create network
net = ICALinear(D_in, D_out)

# create optimizer
opt = torch.optim.SGD(net.parameters(), 1e-6)


for t in range(500):

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = net(x)
    opt.zero_grad()
    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()
    opt.step()
