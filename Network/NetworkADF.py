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
import ADFLayers as ADF
import Layers as Layers
# Don't generate pyc codes
sys.dont_write_bytecode = True


class BasicBlock(tf.keras.Model):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, name_=None, 
        shortCut=True):
        super(BasicBlock, self).__init__()
        self.shortCut = shortCut
            
        self.conv1 = ADF.Conv2d(inChannels, outChannels, kernelSize, 
            stride=stride, name_=name_+"_Conv1", bias=False)
        self.bn1 = ADF.BatchNorm2d(name_=name_+"_bn1")
        self.relu = ADF.ReLU()
        
        self.conv2 = ADF.Conv2d(outChannels, outChannels, kernelSize, 
            name_=name_+"_Conv2", bias=False)
        self.bn2 = ADF.BatchNorm2d(name_=name_+"_bn2")
        
        if shortCut:
            self.conv3 = ADF.Conv2d(inChannels, outChannels, 1, 
                stride=stride, name_=name_+"_Conv3", bias=False)
            self.bn3 = ADF.BatchNorm2d(name_=name_+"_bn3")
        pass    

    def call(self, mean, var):
        net = mean, var
        net1 = self.relu(*self.bn1(*self.conv1(*net)))
        net2 = self.bn2(*self.conv2(*net1))
        if self.shortCut:
            net3 = self.bn3(*self.conv3(*net))
            sumMean = net2[0] + net3[0]
            sumVar = net2[1] + net3[1]
            out = sumMean, sumVar
            return self.relu(*out)
        else:
            sumMean = net[0] + net2[0]
            sumVar = net[1] + net2[1]
            out = sumMean, sumVar
            return self.relu(*out)


class CIFAR_ADF():
    def __init__(self):
        self.number_of_pools = 4
        
        self.conv1 = ADF.Conv2d(3, 64, 3, name_="Conv1", bias=False)
        self.bn = ADF.BatchNorm2d(name_="BN1")
        self.relu = ADF.ReLU()

        self.resBlock1 = BasicBlock(64, 64, 3, stride=1, name_="Residual1", shortCut=False)
        self.resBlock2 = BasicBlock(64, 64, 3, stride=1, name_="Residual2", shortCut=False)
        # self.pool1 = ADF.AvgPool2d()
        self.resBlock3 = BasicBlock(64, 128, 3, stride=2, name_="Residual3", shortCut=True)
        self.resBlock4 = BasicBlock(128, 128, 3, stride=1, name_="Residual4", shortCut=False)

        self.resBlock5 = BasicBlock(128, 256, 3, stride=2, name_="Residual5", shortCut=True)
        self.resBlock6 = BasicBlock(256, 256, 3, stride=1, name_="Residual6", shortCut=False)
        
        self.resBlock7 = BasicBlock(256, 512, 3, stride=2, name_="Residual7", shortCut=True)
        self.resBlock8 = BasicBlock(512, 512, 3, stride=1, name_="Residual8", shortCut=False)

        self.pool = ADF.AvgPool2d()
        numFeatures = (32/(2**(self.number_of_pools)))**2
        self.lin1 = ADF.Linear(512*numFeatures, 10, name_="Linear1")
        # self.drop = ADF.Dropout(p=0.2)
        # self.lin2 = ADF.Linear(512, 10, name_="Linear2")
        self.soft = ADF.Softmax()
        

    def network(self, mean, var):
        # var = tf.zeros_like(mean)+0.001
        # net = [mean, var]
        net = mean, var
        net = self.relu(*self.bn(*self.conv1(*net)))
        net = self.resBlock1(*net)
        net = self.resBlock2(*net)
        # net = self.pool1(*net)
        net = self.resBlock3(*net)
        net = self.resBlock4(*net)

        net = self.resBlock5(*net)
        net = self.resBlock6(*net)
        
        net = self.resBlock7(*net)
        net = self.resBlock8(*net)
        net = self.pool(*net)
        # net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)
        net_mean, net_variance = tf.layers.flatten(net[0]), tf.layers.flatten(net[1])
        net = [net_mean, net_variance]
        net = self.lin1(*net)
        # net = self.drop(net)
        # prLogits = self.lin2(net)
        prLogits = net[0]
        prSoftMax = self.soft(*net) 
        # prSoftMax = tf.nn.softmax(logits = prLogits)
        return prLogits, prSoftMax