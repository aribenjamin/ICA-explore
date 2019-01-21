import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple



class Conv2d_Wrapper(Function):
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
        ctx.save_for_backward(input, weight, bias)
        ctx.nonlinearity_g = nonlinearity_g
        return F.conv2d(input, weight, bias, stride,
                          padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # these checks are here for efficiency.
        # Returning gradients for inputs that don't require them is not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input (* elementwise product*) ctx.nonlinearity_g(input.mm(weight.t()))
            # TODO: check if this gradient aligns with that n the RICA paper
            #TODO implement this as a convolution
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

# Alias the apply method of the above
ica_conv2d = Conv2d_Wrapper.apply

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

    # @weak_script_method
    def forward(self, input):
        return ica_conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups))


    # def __init__(self, in_channels, out_channels, kernel_size, stride=1,
    #              padding=0, output_padding=0, groups=1, bias=True, dilation=1):
    #     kernel_size = _pair(kernel_size)
    #     stride = _pair(stride)
    #     padding = _pair(padding)
    #     dilation = _pair(dilation)
    #     output_padding = _pair(output_padding)
    #     super(ConvTranspose2d, self).__init__(
    #         in_channels, out_channels, kernel_size, stride, padding, dilation,
    #         True, output_padding, groups, bias)
    #
    # @weak_script_method
    # def forward(self, input, output_size=None):
    #     # type: (Tensor, Optional[List[int]]) -> Tensor
    #     output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
    #     return F.conv_transpose2d(
    #         input, self.weight, self.bias, self.stride, self.padding,
    #         output_padding, self.groups, self.dilation)

