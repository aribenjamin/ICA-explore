# ICA as a prior for CNNs

Independent Component Analysis, which seeks to find via a linear transformation
the components of the inputs that are statistically independent as possible, results in filters that look
similar to those observed in the first layer of CNNs. In neuroscience, ICA is both biologically plausible and
can explain V1 simple cell RFs, complex cells, and perhaps V2 and beyond. This raises the question of whether a
good portion of learning in lower layers could be done via ICA instead of just gradient descent. This can be thought
of as a form of regularization, establishing a prior of independent nodes, that will improve generalization if
the prior is close (in a KL sense) to the distribution of networks that work for a task.

## In this repo

* Part 1 *: I investigate the distribution of activations of nodes on a traditional feedforward CNN trained on ImageNet.

* Part 2 *: I train a CNN such that each layer performs ICA. This is done by modifying the 