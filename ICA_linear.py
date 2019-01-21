import torch
from torch.autograd import Variable, Function
import math

import torch
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.nn import init
from torch.nn import Module
#from .._jit_internal import weak_module, weak_script_method # can I leave this off?


class ICA_Linear(Function):
    """This is the autograd function that underlies the new ICA linear module.


    The forward pass does nothing different than Linear;
    The backward pass, however, is modified such that the weights
    learn to make the outputs be the Independent Components (non-Gaussian projections)
    of the inputs.

    The implementation is similar to that of RICA (Le et al., NIPS, 2012): we augment the cost
    function with G(

    Supports finding only super-gaussian source, only sub-gaussian sources, or both"""

    @staticmethod
    def forward(ctx, input, weight, bias=None, super_or_sub = "both", ica_strength = 1e-4, nonlinearity_g = torch.tanh):
        ctx.super_or_sub = super_or_sub
        ctx.ica_strength = ica_strength
        ctx.nonlinearity_g = nonlinearity_g

        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print('Modified backward')
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            raise(NotImplementedError("No ICA grad wrt input yet"))
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            if ctx.super_or_super == "super":
                grad_ica = input*torch.sum(ctx.nonlinearity_g(input.mm(weight.t())))
            elif ctx.super_or_super == "sub":
                grad_ica = -input*torch.sum(ctx.nonlinearity_g(input.mm(weight.t())))
            elif ctx.super_or_super == "both":
                # as in ﻿Lee, Girolami, Sejnowski, 1999, Neural Computation
                s = torch.sign(torch.mean(1/torch.cosh(input)**2)*torch.mean(input**2) -
                               torch.mean(torch.tanh(input)*input))
                grad_ica = s * input*torch.sum(ctx.nonlinearity_g(input.mm(weight.t())))

            print(grad_ica.size(), input.size())
            grad_weight = grad_output.t().mm(input + ctx.ica_strength * grad_ica)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

# Alias the apply method of the above
ica_linear = ICA_Linear.apply

#@weak_module
class ICALinear(Module):

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True,
                 super_or_sub = "super", ica_strength = 1e-4, nonlinearity_g = torch.tanh):
        super(ICALinear, self).__init__()
        self.super_or_sub = super_or_sub
        self.ica_strength = ica_strength
        self.nonlinearity_g = nonlinearity_g
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

#    @weak_script_method
    def forward(self, input):
        return ica_linear(input, self.weight, self.bias,
                          self.super_or_sub, self.ica_strength, self.nonlinearity_g)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

