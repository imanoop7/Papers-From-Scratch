# Alexnet-from-scratch-using-pytorch

## Overview

AlexNet is a convolutional neural network that is 8 layers deep. It was designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton and was the winning entry in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012. AlexNet significantly outperformed the previous state-of-the-art models and is considered a milestone in the field of deep learning.

## Architecture

The AlexNet architecture consists of the following layers:
1. **Input Layer**: The input to AlexNet is a 224x224x3 image.
2. **Convolutional Layer 1**: 96 filters of size 11x11 with a stride of 4 and ReLU activation.
3. **Max-Pooling Layer 1**: Pooling with a 3x3 filter and a stride of 2.
4. **Convolutional Layer 2**: 256 filters of size 5x5 with ReLU activation.
5. **Max-Pooling Layer 2**: Pooling with a 3x3 filter and a stride of 2.
6. **Convolutional Layer 3**: 384 filters of size 3x3 with ReLU activation.
7. **Convolutional Layer 4**: 384 filters of size 3x3 with ReLU activation.
8. **Convolutional Layer 5**: 256 filters of size 3x3 with ReLU activation.
9. **Max-Pooling Layer 3**: Pooling with a 3x3 filter and a stride of 2.
10. **Fully Connected Layer 1**: 4096 neurons with ReLU activation.
11. **Fully Connected Layer 2**: 4096 neurons with ReLU activation.
12. **Output Layer**: 1000 neurons with softmax activation for classification.

## Key Features

- **ReLU Activation**: AlexNet uses the Rectified Linear Unit (ReLU) activation function, which helps in faster training by introducing non-linearity.
- **Dropout**: Dropout is used in the fully connected layers to prevent overfitting.
- **Data Augmentation**: Techniques like image translations, horizontal reflections, and patch extractions are used to artificially increase the size of the training dataset.
- **GPU Utilization**: AlexNet was one of the first models to use GPUs for training, significantly reducing the training time.

## Training

AlexNet was trained on the ImageNet dataset, which contains over 15 million labeled high-resolution images belonging to roughly 22,000 categories. The training process involved using stochastic gradient descent with momentum, data augmentation, and dropout.

## Performance

AlexNet achieved a top-5 error rate of 15.3% in the ILSVRC 2012 competition, which was a significant improvement over the previous state-of-the-art models. This success demonstrated the potential of deep learning and convolutional neural networks in computer vision tasks.


## References

For more details, you can refer to the [original AlexNet paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).
