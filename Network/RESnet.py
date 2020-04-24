"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
   

    #Construct first convolution layer
    net = Img

    net_ = tf.layers.conv2d(inputs = net, name='layer_conv0_1', padding='same',filters = 16, kernel_size = 3, activation = tf.nn.relu)
    net = tf.layers.conv2d(inputs = net_, name='layer_conv1_1', padding='same',filters = 16, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='layer_bn1_1')
    net = tf.nn.relu(net, name="layer_relu_1_1")
    net = tf.layers.conv2d(inputs = net, name='layer_conv2_1', padding='same',filters = 16, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='layer_bn2_1')
    net = tf.nn.relu(net, name="layer_relu_2_1")

    # net = tf.nn.relu(net, name = 'layer_Relu1')
    res_1 = net + net_
    # layer_conv1 = net
    net_ = tf.layers.conv2d(inputs = res_1, name = 'layer_conv0_2', padding= 'same', filters = 32, kernel_size = 3, activation = tf.nn.relu)
    net = tf.layers.conv2d(inputs = net_, name = 'layer_conv1_2', padding= 'same', filters = 32, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net, axis = -1, center = True, scale = True, name ='layer_bn1_2')
    net = tf.nn.relu(net, name="layer_relu_1_2")
    net = tf.layers.conv2d(inputs = net, name = 'layer_conv2_2_', padding= 'same', filters = 32, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn2_2')
    net = tf.nn.relu(net, name="layer_relu_2_2")

    
    res_2 = net + net_
    # layer_conv2 = net
    # net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)
    net  = tf.layers.max_pooling2d(inputs = res_2, pool_size = 2, strides = 2)


    net_ = tf.layers.conv2d(inputs = net, name = 'layer_conv0_3', padding= 'same', filters = 32, kernel_size = 3, activation = tf.nn.relu)
    net = tf.layers.conv2d(inputs = net_, name = 'layer_conv1_3', padding= 'same', filters = 32, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn1_3')
    net = tf.nn.relu(net, name="layer_relu_1_3")
    net = tf.layers.conv2d(inputs = net, name = 'layer_conv2_3', padding= 'same', filters = 32, kernel_size = 3, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn2_3')
    net = tf.nn.relu(net, name="layer_relu_2_3")
    res_3 = net + net_
    
    res_3  = tf.layers.max_pooling2d(inputs = res_3, pool_size = 2, strides = 2)

    # net_ = tf.layers.conv2d(inputs = res_3, name = 'layer_conv4', padding= 'same', filters = 16, kernel_size = 3, activation = tf.nn.relu)
    # net = tf.layers.conv2d(inputs = net_, name = 'layer_conv4_', padding= 'same', filters = 512, kernel_size = 3, activation = tf.nn.relu)
    # net = tf.layers.conv2d(inputs = net, name = 'layer_conv4__', padding= 'same', filters = 512, kernel_size = 3, activation = tf.nn.relu)
    # net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn4')

    # net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = tf.layers.flatten(net_)

    #Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 512, activation = tf.nn.relu)
    net = tf.layers.dense(inputs = net, name ='layer_fc2',units=256, activation = tf.nn.relu)
    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = 10, activation = None)
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)
    return prLogits, prSoftMax

