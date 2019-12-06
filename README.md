# CNN

# Part 1

We started with a basic 1D cnn and RNN structure. We used an embedding layer, a conv1D layer, a max1d layer, a simple RNN layer, a dropout layer, and a dense, and sigmoid 1 node output layer. All our default activation functions were ReLu because those functions are efficient to use. We started with a kernel size of 3. On 4 epochs this gave an accuracy of 77.6%. Changing the kernel size to 5 gave an accuracy of 80%.  Changing the kernel size to 10 gave an accuracy of 76.7%. Changing the kernel size to 7 gave an accuracy of 85%. Changing kernel size to 8 gave an accuracy of 70%. Changing kernel size to 6 gave an accuracy of 75%. Re-running with a kernel size of 7 gave an accuracy of 84%.

Then I changed the pooling size from 2-3. This gave an accuracy of 93% on the training set, but 76% on the testing set. The model clearly overfit, so we decided to add regularization. I decided to add a dropout of .3 between the CNN and RNN layers, and re-ran the model. This model resulted in an accuracy of 78% on the training set, and 84% on the testing set. The problem here was that the model did not train for enough epochs - it was still gaining significant accuracy. I re-ran the model here for 10 epochs and got 80.5% on the testing set. Interestingly, epoch 9 and 10 had accuracy of .89 and .76 respectively. It is interesting what happened in the training process here. I went ahead and increase max pooling size to 4 and re-ran the model. This model received an accuracy of 84%. Upping the pooling size to 5 only achieved 83% accuracy, therefore we deciding to take the next step and upgrade to a LSTM layer. We also changed the pooling size back to 4. This model achieved 75% accuracy on testing set and 83% on training set. We removed the LSTM layer and increased the # of filters to 32 and ran the model. Increasing filter to 32 gave us 87% accuracy on testing set (results are pickled in "CNN-RNN-1")

# Part 2

We started with a base vanilla CNN model with a conv2d layer followed by a maxpooling layer, repeated 3 times.

We ran the model for 50 epochs and got 86% accuracy on the training set and 75.6% accuracy on the test set (results are pickled in "CNN-CIFAR"). Since the barrier seemed to be regularization, we decided to then increase regularization (increased drop out rate from .2 to .4 in dropout layers occurring after the conv2D and maxpooling layers) to try to get the remaining 4% accuracy. We also increased epochs to 100 to compensate for the increase in required training due to the increase in regularization. We ran the model for 100 epochs and got 77.18% accuracy on the training set and 75.97% accuracy on the test set (results are pickled in "CNN-CIFAR-1-ANDRE"). We noticed that there was not much increase in accuracy after 50 epochs for this model for the training set. Therefore, we decided to run a model with .3 drop out at 20 epochs. We had the understanding that .4 dropout may be unnecessary, and 100 epochs may be too large. When we changed parameters to .3 dropout at 20 epochs we got 81.9% accuracy on the training set and 78.6% accuracy on the test set (results are pickled in "CNN-CIFAR-1-GEORGE"). We decided to change the dropout rate to .2 after our CNN layers and .3 in the dense layer. This model resulted in an accuracy of 87.42% on the training data and 79.85% in testing data (results are pickled in "CNN-CIFAR-2-George"). We decided to up the epochs from 50 to 100 and we received an accuracy of 89.93% on the training data and 80.93% in testing data (results are pickled in "CNN-CIFAR-3-George"). We tried increasing batch size from 64 to 128 and received an accuracy of 88.07% on the training set and 80.78% on the testing set (results are pickled in "CNN-CIFAR-2-ANDRE"). To see if we could break through even higher accuracy we tried implementing data augmentation into our model. We ran this model for 50 epochs and received a training set accuracy of 80.4%. We decided to ditch this model since it provided no significant benefit over our previous model.. but we did consider data augmentation. Future work could look into implementing data augmentation in different ways to see if additional accuracy could be obtained.
