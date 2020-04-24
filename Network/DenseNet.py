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


    net_1_1 = tf.layers.conv2d(inputs = net, padding='same',filters = 16, kernel_size = 3, activation = tf.nn.relu)
    net_1_2 = tf.layers.conv2d(inputs = net_1_1, padding='same',filters = 16, kernel_size = 3, activation = tf.nn.relu)
    net_1_3 = tf.layers.conv2d(inputs = net_1_1 + net_1_2, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)
    net_1_4 = tf.layers.conv2d(inputs = net_1_1 + net_1_2 + net_1_3, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)
    net_1_5 = tf.layers.conv2d(inputs = net_1_1 + net_1_2 + net_1_3 + net_1_4, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)

    net  = tf.layers.max_pooling2d(inputs = net_1_5, pool_size = 2, strides = 2)


    net_2_1 = tf.layers.conv2d(inputs = net, padding='same',filters = 16, kernel_size = 3, activation = tf.nn.relu)
    net_2_2 = tf.layers.conv2d(inputs = net_2_1, padding='same',filters = 16, kernel_size = 3, activation = tf.nn.relu)
    net_2_3 = tf.layers.conv2d(inputs = net_2_1 + net_2_2, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)
    net_2_4 = tf.layers.conv2d(inputs = net_2_1 + net_2_2 + net_2_3, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)
    net_2_5 = tf.layers.conv2d(inputs = net_2_1 + net_2_2 + net_2_3 + net_2_4, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)

    net  = tf.layers.max_pooling2d(inputs = net_2_5, pool_size = 2, strides = 2)

    # net_3_1 = tf.layers.conv2d(inputs = net, padding='same',filters = 16, kernel_size = 3, activation = tf.nn.relu)
    # net_3_2 = tf.layers.conv2d(inputs = net_3_1, padding='same',filters = 16, kernel_size = 3, activation = tf.nn.relu)
    # net_3_3 = tf.layers.conv2d(inputs = net_3_1 + net_3_2, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)
    # net_3_4 = tf.layers.conv2d(inputs = net_3_1 + net_3_2 + net_3_3, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)
    # net_3_5 = tf.layers.conv2d(inputs = net_3_1 + net_3_2 + net_3_3 + net_3_4, padding='same', filters=16, kernel_size=3, activation=tf.nn.relu)

    # net  = tf.layers.max_pooling2d(inputs = net_3_5, pool_size = 2, strides = 2)

    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 512, activation = tf.nn.relu)
    net = tf.layers.dense(inputs = net, name ='layer_fc2',units=256, activation = tf.nn.relu)
    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = 10, activation = None)
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)
    return prLogits, prSoftMax

