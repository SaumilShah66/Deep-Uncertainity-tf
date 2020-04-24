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
import trials as ADF
# Don't generate pyc codes
sys.dont_write_bytecode = True



class CIFARNormal():
    def __init__(self):
        self.conv_1 = ADF.Conv2d_(3, 16, 3, name_="conv_1")
        self.lin2 = ADF.Linear_(16*16*16, 128, name_="Lin1")
        self.lin3 = ADF.Linear_(128, 10, name_="lin2")
        self.soft = ADF.Softmax_()
    
    def network(self, mean):
        net = mean
        net = self.conv_1(net)
        net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)
        
        net = tf.layers.flatten(net)
        
        net = self.lin2(net)
        net = self.lin3(net)

        prLogits = net 

        # prSoftMax = tf.nn.softmax(logits = prLogits)
        return prLogits, prSoftMax

class CIFAR10Model():
    def __init__(self):
        self.conv_1 = ADF.Conv2d(3, 16, 3, name_="conv_1")
        # self.conv_2 = ADF.Conv2d(16, 16, 3, name_="conv_2")
        # self.conv_3 = ADF.Conv2d(16, 32, 3, name_="Conv_3")
        # self.conv_4 = ADF.Conv2d(32, 32, 3, name_="Conv_4")
        self.pool_1 = ADF.MaxPool2d()
        # self.conv_5 = ADF.Conv2d(32, 32, 3, name_="Conv_5")
        # self.conv_6 = ADF.Conv2d(32, 32, 3, name_="conv_6")
        # self.pool_2 = ADF.MaxPool2d()

        # self.lin1 = ADF.Linear(name_="lin_1")
        self.lin2 = ADF.Linear(16*16*16, 128, name_="Lin1")
        self.lin3 = ADF.Linear(128, 10, name_="lin2")
        self.soft = ADF.Softmax()

    def network(self, mean, variance):
        net = mean, variance
        net = self.conv_1(*net)
        # net = self.conv_2(*net)
        # net = self.conv_3(*net)
        # net = self.conv_4(*net)
        net = self.pool_1(*net)
        # net = self.conv_5(*net)
        # net = self.conv_6(*net)
        # net = self.pool_2(*net)

        net1_m = tf.layers.flatten(net[0])
        net1_v = tf.layers.flatten(net[1])

        net = net1_m, net1_v
        # net = self.lin1(net[0], net[1], 256)
        net = self.lin2(*net)
        net = self.lin3(*net)

        prLogits = net[0]
        prSoftMax = self.soft(*net)
        return prLogits, prSoftMax[0]

