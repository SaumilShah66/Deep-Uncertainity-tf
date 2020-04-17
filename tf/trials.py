import tensorflow as tf
import numpy as np

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


