import torch
from torch.autograd import gradcheck
from ICA_linear import ICALinear

# gradcheck takes a tuple of tensor as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)
test = gradcheck(ICALinear.apply, input, eps=1e-6, atol=1e-4)
print(test)