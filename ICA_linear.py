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

    Supports finding only super-gaussian source, only sub-gaussian sources, or both."""

    @staticmethod
    def forward(ctx, input, weight, bias=None, super_or_sub = "both", ica_strength = 1e-4, nonlinearity_g = torch.tanh):
        ctx.super_or_sub = super_or_sub
        ctx.ica_strength = ica_strength
        ctx.nonlinearity_g = nonlinearity_g

        ## TODO this is a test
        # Test: does this work if I wrap the linear function?
#        output = input.mm(weight.t())

#        if bias is not None:
#            output += bias.unsqueeze(0).expand_as(output)

        output = torch.nn.functional.linear(input,weight,bias)

        ctx.save_for_backward(input, weight, bias, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print('Modified backward')
        input, weight, bias, output = ctx.saved_variables
        grad_input = grad_weight = grad_bias = grad_super_or_sub = grad_ica_strength = grad_nonlinearity_g = None

        if ctx.needs_input_grad[0]:
            raise(NotImplementedError("No ICA grad wrt input yet; question of whether we want to pass g_ICA down"))
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            if ctx.super_or_sub == "super":
                grad_ica = ctx.nonlinearity_g(output).t().mm(input)
            elif ctx.super_or_sub == "sub":
                grad_ica = -ctx.nonlinearity_g(output).t().mm(input)
            elif ctx.super_or_sub == "both":
                # as in Lee, Girolami, Sejnowski, 1999, Neural Computation
#                 s = torch.sign(torch.sum(torch.reciprocal(torch.cosh(output)**2),0)*torch.sum(output**2,0) -
#                                torch.sum(torch.tanh(output)*output,0)).view(-1,1)
                
                s  = torch.tensor([ -1.,  -1., -1.,  -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                         1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                         1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]).view(-1,1).cuda()
                grad_ica = s * ctx.nonlinearity_g(output).t().mm(input)
            elif ctx.super_or_sub == "fastICA":
                # WARNING implementation could be incorrect
                beta = torch.sum(output * ctx.nonlinearity_g(output),0)
                Dp = torch.reciprocal(beta - torch.sum(1-torch.tanh(output)**2,0))
                D = torch.diag(Dp)
                middle = torch.diag(-beta) + ctx.nonlinearity_g(output).t().mm(output)
#                 print("output",output.size(),"beta",beta.size(),"Dp",Dp.size(),"D",D.size(),"middle",middle.size())
                grad_ica = D.mm(middle).mm(weight)
                
                
                
            ## Confusing note: right now I'm having to normalize by the batch size
            ## and also by the output dimension in order to get gradient values
            ## that agree with those calculated via backpropagation upon the ICA cost
            ## directly. To be resolved.
            bs_times_output_dim = grad_output.size()[0]*grad_output.size()[1]


            ## TODO this is a test
#            grad_weight = grad_output.t().mm(input) + ctx.ica_strength * grad_ica / bs_times_output_dim
            grad_weight = ctx.ica_strength * grad_ica/ bs_times_output_dim

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
#        grad_input = grad_weight = grad_bias = grad_super_or_sub = grad_ica_strength = grad_nonlinearity_g = None

        return grad_input, grad_weight, grad_bias, grad_super_or_sub, grad_ica_strength, grad_nonlinearity_g

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

