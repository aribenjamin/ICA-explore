import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple



class ICA_Conv2d(Function):
    """This is the autograd function that underlies the 2d convoltiion + ICA module (Ica_Conv2d).

    The forward pass does nothing different than Conv2d; this just wraps the F.conv2d function.
    The backward pass, however, is modified such that the weights of the wrapped Conv2d
    learn to make the Conv2d outputs be the Independent Components (non-Gaussian projections)
    of the Conv2d inputs.

    This function uses the RICA conception of ICA, which takes the effective cost of
    C = sum(x - W W.T x) + E(sum(G[Wx])). For a single output, the derivative of the non-Gaussianity component is
    the nonlinear Hebbian rule E(x g(w.x)) where g = G'. In matrix form,
    for many components, we have dC/dW = E(x g(W))"""

    @staticmethod
    def forward(ctx, input, weight, bias, stride,
                    padding, dilation, groups, nonlinearity_g = torch.tanh):

        output = F.conv2d(input, weight, bias, stride,
                          padding, dilation, groups)
        ctx.save_for_backward(input, weight, bias, output)
        ctx.nonlinearity_g = nonlinearity_g
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias, output = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        grad_stride = grad_padding = grad_dilation = grad_groups = grad_nonlinearity_g = None

        # an easy way to get the gradient wrt the non-gaussianity is to calculate the
        # nongaussianity, then just call backwards on this.
        # From https://github.com/pytorch/pytorch/issues/1776
        # (but perhaps isn't maximally efficient; one could calculate the derivative manually)

        nongaussianity = torch.mean(torch.log(torch.cosh(output)))

        if ctx.needs_input_grad[0]:
            grad_input = torch.autograd.grad(nongaussianity, input, grad_output)
        if ctx.needs_input_grad[1]:
            if ctx.super_or_sub == "super":
                grad_weight = ctx.ica_strength *torch.autograd.grad(nongaussianity, weight, grad_output)
            elif ctx.super_or_sub == "sub":
                grad_weight = -ctx.ica_strength *torch.autograd.grad(nongaussianity, weight, grad_output)

        # no change in bias gradient

        return grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups, grad_nonlinearity_g

# Alias the apply method of the above
ica_conv2d = ICA_Conv2d.apply

#@weak_module
class Ica_Conv2d(_ConvNd):
    """This module is like a Conv2d Module, with the only difference being that there is a prior
    set upon the weights such that the convolution outputs are Independent Components (non-Gaussian projections)
    of the inputs to the filter."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Ica_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.ica_conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)