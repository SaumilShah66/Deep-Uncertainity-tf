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

mean = torch.tensor([[1,  2,  3,  4],
					  [5,  6,  7,  8],
					  [9, 10, 11, 12],
					  [13, 14, 15, 16]], dtype=torch.float)
variance = torch.zeros_like(mean) + 0.001
kernel_size = 2
mean = mean.reshape(1,1,4,4)
variance = variance.reshape(1,1,4,4)
# AvgPooledTest(mean, variance, kernel_size)
MaxPool2dTest(mean, variance)