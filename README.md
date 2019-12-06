# CNN

# Part 1

# Part 2

We started with a base vanilla CNN model with a conv2d layer followed by a maxpooling layer, repeated 3 times.

We ran the model for 50 epochs and got 86% accuracy on the training set and 75.6% accuracy on the test set (results are pickled in "CNN-CIFAR"). Since the barrier seemed to be regularization, we decided to then increase regularization (increased drop out rate from .2 to .4 in dropout layers occurring after the conv2D and maxpooling layers) to try to get the remaining 4% accuracy. We also increased epochs to 100 to compensate for the increase in required training due to the increase in regularization. We ran the model for 100 epochs and got 77.18% accuracy on the training set and 75.97% accuracy on the test set (results are pickled in "CNN-CIFAR-1-ANDRE"). We noticed that there was not much increase in accuracy after 50 epochs for this model for the training set. Therefore, we decided to run a model with .3 drop out at 20 epochs. We had the understanding that .4 dropout may be unnecessary, and 100 epochs may be too large. When we changed parameters to .3 dropout at 20 epochs we got 81.9% accuracy on the training set and 78.6% accuracy on the test set.
