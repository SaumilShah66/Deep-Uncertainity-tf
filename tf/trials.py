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

		
	def call(self, inputs_mean, inputs_variance, pool_size):
		pool_layer = tf.keras.layers.AvgPool2D(pool_size=pool_size)
		outputs_mean = pool_layer(inputs_mean)
		outputs_variance = pool_layer(inputs_variance)
		shape = inputs_mean.shape.as_list()
		outputs_variance = outputs_variance/(shape[1]*shape[2])
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance


# class AvgPool2d(nn.Module):
#     def __init__(self, keep_variance_fn=None):
#         super(AvgPool2d, self).__init__()
#         self._keep_variance_fn = keep_variance_fn
		
#     def forward(self, inputs_mean, inputs_variance, kernel_size):
#         outputs_mean = F.avg_pool2d(inputs_mean, kernel_size)
#         outputs_variance = F.avg_pool2d(inputs_variance, kernel_size)
#         outputs_variance = outputs_variance/(inputs_mean.size(2)*inputs_mean.size(3))
		
#         if self._keep_variance_fn is not None:
#             outputs_variance = self._keep_variance_fn(outputs_variance)
		
#         return outputs_mean, outputs_variance