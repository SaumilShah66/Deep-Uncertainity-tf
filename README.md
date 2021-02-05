# Deep Learning Uncertainty Prediction-TensorFlow

Implemented the paper [A General Framework for Uncertainty Estimation in Deep Learning](http://rpg.ifi.uzh.ch/docs/RAL20_Loquercio.pdf) in Tensorflow. Custom layers are defiend using base class of __tf.keras.Model__ to define the Assumed Density  Filtering (ADF) layers.<br>

ADF layers take 2 inputs [mean, variance]. Here __mean__ refers to the input training data. __variance__ is randomly initialized and is then trained, Variance is dependant on square of network weights (of training data). Mean and Variance affect each other in the MaxPool, ReLU and Conv2D layers.<br>

As per the [paper](http://rpg.ifi.uzh.ch/docs/RAL20_Loquercio.pdf), the proposed method is not mathematically stable and may lead to __NAN__ outputs. Hence initial training is to be done with a regular model (non-adf), and then to be __transfer learning to be initiated with ADF__.<br>

In order to copy weights, similarity in models is must. Hence regular framework layers (single input) were also custom defiend in the saame way as ADF's mean layers. 

## Authors
- Saumil Shah
- Varun Asthana

## Dual input layers implemented
Implementation of custom layers with 2 inputs have been completed for-
- [x] Maxpool2d
- [x] AvePool2d
- [x] Softmax
- [x] ReLU
- [x] LeakyReLU
- [x] Dropout
- [x] Linear
- [x] BatchNorm2d
- [x] Conv2d
- [x] ConvTranspose2d
- [x] concatenate_as
- [x] Sequential

## Current
- Train RESNet model for cifar10
- Train RESNet model for cifar10 on ADF

## Result
We were able to reproduce the results on CIFAR10 dataset from the original paper Table III.

<p align="center">
<img src="https://github.com/SaumilShah66/Deep-Uncertainity-tf/blob/ModelDev/images/table3.png" width = 500>
</p>


## Sample Testing

The below image shows a real good example of the use of this model. This image of frog was one of the top 5% error. Without the proposed model, ResNet18 model predicts it as a dog with high confidence, but ResNet18 with ADF layers gives all the predictions below the probability of 0.4, which says that model is quite uncertain about what the image is.

<p align="center">
<img src="https://github.com/SaumilShah66/Deep-Uncertainity-tf/blob/ModelDev/images/frog.png" width = 500>
</p>

## Download
```
$ git clone https://github.com/SaumilShah66/Deep-Uncertainity-tf.git
```
Download CIFAR10 dataset
```
$ git clone https://github.com/varunasthana92/CIFAR10.git
```

## Reference
Reference : https://github.com/uzh-rpg/deep_uncertainty_estimation