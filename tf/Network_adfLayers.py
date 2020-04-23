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
import adfLayers as ADF
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def keep_variance(x, min_variance):
    return x + min_variance

class CIFAR10Model():
    def __init__(self):
        self.keep_variance_fn = lambda x: keep_variance(x, min_variance= 0.001)
        self.conv_1 = ADF.Conv2d(3, 16, 3, keep_variance_fn= self.keep_variance_fn, name_="conv_1")
        self.batch_norm_1 = ADF.BatchNorm2d(keep_variance_fn= self.keep_variance_fn)
        self.relu_1 = ADF.ReLU(keep_variance_fn= self.keep_variance_fn)
        # self.conv_2 = ADF.Conv2d(16, 16, 3, name_="conv_2")
        # self.conv_3 = ADF.Conv2d(16, 32, 3, name_="Conv_3")
        # self.conv_4 = ADF.Conv2d(32, 32, 3, name_="Conv_4")
        self.pool_1 = ADF.MaxPool2d()
        # self.conv_5 = ADF.Conv2d(32, 32, 3, name_="Conv_5")
        # self.conv_6 = ADF.Conv2d(32, 32, 3, name_="conv_6")
        # self.pool_2 = ADF.MaxPool2d()

        # self.lin1 = ADF.Linear(name_="lin_1")
        # self.lin2 = ADF.Linear(name_="lin_2")
        self.lin3 = ADF.Linear(in_features=4096, out_features=10, keep_variance_fn= self.keep_variance_fn, name_="lin_3")
        self.soft = ADF.Softmax()

    def network(self, mean, variance):
        net = mean, variance
        net = self.conv_1(*net)
        net = self.batch_norm_1(*net)
        net = self.relu_1(*net)
        # net = self.conv_2(*net)
        # net = self.conv_3(*net)
        # net = self.conv_4(*net)
        net = self.pool_1(*net)
        # net = self.conv_5(*net)
        # net = self.conv_6(*net)
        # net = self.pool_2(*net)

        net1_m = tf.layers.flatten(net[0])
        net1_v = tf.layers.flatten(net[1])
        print('net 0 shape= ', net1_m.shape)
        net = net1_m, net1_v
        # net = self.lin1(net[0], net[1], 256)
        # net = self.lin2(net[0], net[1], 128)
        net = self.lin3(*net)

        prLogits = net[0]
        prSoftMax = self.soft(*net)
        return prLogits, prSoftMax[0]

'''
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
        # self.lin2 = ADF.Linear(name_="lin_2")
        self.lin3 = ADF.Linear(name_="lin_3")
        self.soft = ADF.Softmax()

    def network(self, mean, variance):
        net = mean, variance
        # net = self.conv_1(*net)
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
        # net = self.lin2(net[0], net[1], 128)
        net = self.lin3(net[0], net[1], 10)

        prLogits = net[0]
        prSoftMax = self.soft(*net)
        return prLogits, prSoftMax[0]
'''