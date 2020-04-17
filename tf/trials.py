import tensorflow as tf
import numpy as np
from numbers import Number

def test(self, block, num_blocks, num_classes=10, noise_variance=1e-3, min_variance=1e-3, initialize_msra=False):
    self.keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
    self._noise_variance = noise_variance
    self.in_planes = 64
    return self.keep_variance_fn

def normcdf(value, mu=0.0, stddev=1.0):
    sinv = (1.0 / stddev) if isinstance(stddev, Number) else tf.reciprocal(stddev)
    return 0.5 * (1.0 + tf.erf((value - mu) * sinv / tf.sqrt(2.0)))

def _normal_log_pdf(value, mu, stddev):
    var = (stddev ** 2)
    log_scale = tf.log(stddev) if isinstance(stddev, Number) else tf.log(stddev)
    return -((value - mu) ** 2) / (2.0*var) - log_scale - tf.log(tf.sqrt(2.0*tf.pi))


# Tested against Matlab: Works correctly!
def normpdf(value, mu=0.0, stddev=1.0):
    return tf.exp(_normal_log_pdf(value, mu, stddev))

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
        mu_a = input_mean[:, :, :, 0::2]
        mu_b = input_mean[:, :, :, 1::2]
        var_a = input_variance[:, :, :, 0::2]
        var_b = input_variance[:, :, :, 1::2]
        output_mean, output_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return output_mean, output_variance

    def _max_pool_2x1(self, input_mean, input_variance):
        mu_a = input_mean[:, :, 0::2, :]
        mu_b = input_mean[:, :, 1::2, :]
        var_a = input_variance[:, :, 0::2, :]
        var_b = input_variance[:, :, 1::2, :]
        output_mean, output_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return output_mean, output_variance

    def call(self, input_mean, input_variance):
        z_mean, z_variance = self._max_pool_1x2(input_mean, input_variance)
        output_mean, output_variance = self._max_pool_2x1(z_mean, z_variance)
        return output_mean, output_variance



