#!/usr/bin/env python

import torch
import pandas as pd
import torchvision.models as models
from torchvision import transforms, datasets
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--hist_n_bins', type=int, default=2000)
parser.add_argument('--hist_maxval', type=int, default=20)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--model', type=str, default='vgg',
                        choices=('vgg', 'resnet', 'densenet'))
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(args.device)
n_bins = args.hist_n_bins
maxval = args.hist_maxval
verbose = args.verbose


histograms = {}

def bin_activations_hook_conv2d(module, input, output):
    """
    Keeps a running histogram of the activations of each channel.
    Bin over batch, but also width and height of the image. """
    s = output.size()

    name = module.__my_repr__
    if not name in histograms:
        if verbose>1:
            print("Added hist for {}".format(name))
        histograms[name] = torch.zeros(n_bins)
        for channel in range(s[1]):
            histograms[name + "_" + str(channel)] = torch.zeros(n_bins)

    # has to loop over channels since histc has no axis option
    cpu_output = output.cpu()
    for channel in range(s[1]):
        binned = torch.histc(cpu_output[:,channel,:,:],bins = n_bins, min = -maxval, max = maxval)
        #add inplace
        histograms[name+ "_" + str(channel)].add_(binned)

## Load the model
if verbose > 0:
    print("Loading {}...".format(args.model))
if args.model == 'vgg':
    model = models.vgg16(pretrained=True).cuda()
elif args.model == 'resnet':
    model = models.resnet18(pretrained=True).cuda()
elif args.model == 'densenet':
    model = models.densenet161(pretrained=True).cuda()
else:
    raise
    #
    # model = models.alexnet(pretrained=True).cuda()
    # model = models.squeezenet1_0(pretrained=True).cuda()
    # model = models.inception_v3(pretrained=True).cuda()

idx = 0
for inner_mod in list(model.modules()):
    if verbose > 1:
        print(inner_mod.__class__.__name__)
    if inner_mod.__class__.__name__ == 'Conv2d':
        inner_mod.__my_repr__ = 'Conv2d'+str(idx)
        idx+=1
        if verbose > 1:
            print(' ^ hooked')
        inner_mod.register_forward_hook(bin_activations_hook_conv2d)



## Load the data
if verbose > 0:
    print("Running through images...")
valdir = '/data2/imagenet/val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=args.bs, shuffle=False,
    num_workers=4, pin_memory=True)

### iterate through the data

# switch to evaluate mode
model.eval()

for i, (input_var, _) in enumerate(val_loader):
    with torch.no_grad():
        if verbose>1:
            print("Batch {}".format(i))
        input_var = torch.autograd.Variable(input_var).cuda()

        # compute features, one per layer
        out = model(input_var)


# divide the histograms by the number of batches
for mod in histograms:
    histograms[mod] = histograms[mod].div(float(i))

# save features
all_features = pd.DataFrame(histograms)
all_features.to_hdf('feature_histograms_{}.h5'.format(args.model),
                    key = 'yep', mode='w')
