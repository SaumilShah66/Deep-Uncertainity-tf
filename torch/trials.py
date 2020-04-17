import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair
from contrib.math import normpdf, normcdf

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
            
        return outputs_mean, outputs_variance
    
class MaxPool2d(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(MaxPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def _max_pool_internal(self, mu_a, mu_b, var_a, var_b):
        stddev = torch.sqrt(var_a + var_b)
        ab = mu_a - mu_b
        alpha = ab / stddev
        pdf = normpdf(alpha)
        cdf = normcdf(alpha)
        z_mu = stddev * pdf + ab * cdf + mu_b
        z_var = ((mu_a + mu_b) * stddev * pdf +
                 (mu_a ** 2 + var_a) * cdf +
                 (mu_b ** 2 + var_b) * (1.0 - cdf) - z_mu ** 2)
        if self._keep_variance_fn is not None:
            z_var = self._keep_variance_fn(z_var)
        return z_mu, z_var

    def _max_pool_1x2(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, :, 0::2]
        mu_b = inputs_mean[:, :, :, 1::2]
        var_a = inputs_variance[:, :, :, 0::2]
        var_b = inputs_variance[:, :, :, 1::2]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def _max_pool_2x1(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, 0::2, :]
        mu_b = inputs_mean[:, :, 1::2, :]
        var_a = inputs_variance[:, :, 0::2, :]
        var_b = inputs_variance[:, :, 1::2, :]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def forward(self, inputs_mean, inputs_variance):
        z_mean, z_variance = self._max_pool_1x2(inputs_mean, inputs_variance)
        outputs_mean, outputs_variance = self._max_pool_2x1(z_mean, z_variance)
        return outputs_mean, outputs_variance



# if __name__ == "__main__":