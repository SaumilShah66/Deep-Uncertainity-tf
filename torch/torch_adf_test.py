import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair
from trials import *

def AvgPooledTest(mean, variance, kernel_size):
	avgPool1 = AvgPool2d()
	l = avgPool1(mean, variance, kernel_size)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])
	pass


def MaxPool2dTest(mean, variance):
	maxPool2d = MaxPool2d()
	l = maxPool2d(mean, variance)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])	
	pass

def ReLUTest(mean, variance):
	relu = ReLU()
	l = relu(mean, variance)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])
	pass

def LeakyReLUTest(mean, variance):
	leakyrelu = LeakyReLU()
	l = leakyrelu(mean, variance)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])
	pass

def DropoutTest(mean, variance):
	drop = Dropout()
	l = drop(mean, variance)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])
	pass

def ConvTranspose2dTest(mean, variance):
	deconv = ConvTranspose2d(1, 1, 3)
	l = deconv(mean, variance)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])
	print(l[0].shape)
	pass


def BatchNorm2dTest(mean, variance):
	deconv = BatchNorm2d(1)
	l = deconv(mean, variance)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])
	print(l[0].shape)
	pass

def softmaxTest(mean, variance):
	soft = Softmax(2)
	l = soft(mean, variance)
	print("Mean " + "-"*20)
	print(l[0])
	print("Variance " + "-"*20)
	print(l[1])
	print(l[0].shape)
	pass
mean = torch.tensor([[1,  2,  3,  4],
					  [5,  6,  7,  8],
					  [9, 10, 11, 12],
					  [13, 14, 15, 16]], dtype=torch.float)

kernel_size = 2
mean = mean.reshape(1,1,4,4)
variance = torch.zeros_like(mean) + 0.001
# variance = variance.reshape(1,1,4,4)
# AvgPooledTest(mean, variance, kernel_size)
# MaxPool2dTest(mean, variance)
# ReLUTest(mean, variance)
# LeakyReLUTest(mean, variance)
# DropoutTest(mean, variance)
# ConvTranspose2dTest(mean, variance)
# BatchNorm2dTest(mean, variance)
softmaxTest(mean, variance)
