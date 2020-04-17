import tensorflow as tf
import numpy as np
from numbers import Number
from maths import normpdf, normcdf

def test(self, block, num_blocks, num_classes=10, noise_variance=1e-3, min_variance=1e-3, initialize_msra=False):
    self.keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
    self._noise_variance = noise_variance
    self.in_planes = 64
    return self.keep_variance_fn

class AvgPool2d(tf.keras.Model):
    def __init__(self, keep_variance_fn=None):
        super(AvgPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def call(self, input_mean, inputs_variance, pool_size):
        pool_layer = tf.keras.layers.AvgPool2D(pool_size=pool_size)
        output_mean = pool_layer(input_mean)
        output_variance = pool_layer(inputs_variance)
        shape = input_mean.shape.as_list()
        output_variance = output_variance/(shape[1]*shape[2])
        if self._keep_variance_fn is not None:
            output_variance = self._keep_variance_fn(output_variance)
        return output_mean, output_variance

class MaxPool2d(tf.keras.Model):
    def __init__(self, keep_variance_fn=None):
        super(MaxPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def _max_pool_internal(self, mu_a, mu_b, var_a, var_b):
        stddev = tf.sqrt(var_a + var_b)
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

    def _max_pool_1x2(self, input_mean, input_variance):
        mu_a = input_mean[:, :,0::2,:]
        mu_b = input_mean[:, :,1::2, :]
        var_a = input_variance[:, :, 0::2, :]
        var_b = input_variance[:, :, 1::2, :]
        output_mean, output_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return output_mean, output_variance

    def _max_pool_2x1(self, input_mean, input_variance):
        mu_a = input_mean[:, 0::2, :, :]
        mu_b = input_mean[:, 1::2, :, :]
        var_a = input_variance[:, 0::2, :, :]
        var_b = input_variance[:, 1::2, :, :]
        output_mean, output_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return output_mean, output_variance

    def call(self, input_mean, input_variance):
        z_mean, z_variance = self._max_pool_1x2(input_mean, input_variance)
        output_mean, output_variance = self._max_pool_2x1(z_mean, z_variance)
        return output_mean, output_variance

class ReLU(tf.keras.Model):
    def __init__(self, keep_variance_fn=None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def call(self, feature_mean, feature_variance):
        feature_stddev = tf.cast(tf.sqrt(feature_variance), tf.float64)
        div = feature_mean / feature_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        output_mean = feature_mean * cdf + feature_stddev * pdf
        output_variance = (feature_mean ** 2 + feature_variance) * cdf \
                           + feature_mean * feature_stddev * pdf - output_mean ** 2
        if self._keep_variance_fn is not None:
            output_variance = self._keep_variance_fn(output_variance)
        return output_mean, output_variance



