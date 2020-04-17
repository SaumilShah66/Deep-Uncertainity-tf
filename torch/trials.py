import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair


def keep_variance(x, min_variance):
    return x + min_variance

class AvgPool2d(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(AvgPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        
    def forward(self, inputs_mean, inputs_variance, kernel_size):
        outputs_mean = F.avg_pool2d(inputs_mean, kernel_size)
        outputs_variance = F.avg_pool2d(inputs_variance, kernel_size)
        outputs_variance = outputs_variance/(inputs_mean.size(2)*inputs_mean.size(3))
        
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
            
        # TODO: avg pooling means that every neuron is multiplied by the same 
        #       weight, that is 1/number of neurons in the channel 
        #      outputs_variance*1/(H*W) should be enough already
        
        return outputs_mean, outputs_variance
    

def AvgPooledTest(mean, variance, kernel_size):
	avgPool1 = AvgPool2d()
	l = avgPool1(mean, variance, kernel_size)
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
AvgPooledTest(mean, variance, kernel_size)

# if __name__ == "__main__":