import tensorflow as tf
import numpy as np
from numbers import Number
from maths import normpdf, normcdf
# from tensorflow.python.keras import backend as K

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

class LeakyReLU(tf.keras.Model):
	def __init__(self, negative_slope=0.01, keep_variance_fn=None):
		super(LeakyReLU, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self._negative_slope = negative_slope

	def call(self, features_mean, features_variance):
		features_stddev = tf.cast(tf.sqrt(features_variance), tf.float64)
		div = features_mean / features_stddev
		pdf = normpdf(div)
		cdf = normcdf(div)
		negative_cdf = 1.0 - cdf
		mu_cdf = features_mean * cdf
		stddev_pdf = features_stddev * pdf
		squared_mean_variance = features_mean ** 2 + features_variance
		mean_stddev_pdf = features_mean * stddev_pdf
		mean_r = mu_cdf + stddev_pdf
		variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r ** 2
		mean_n = - features_mean * negative_cdf + stddev_pdf
		variance_n = squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n ** 2
		covxy = - mean_r * mean_n
		outputs_mean = mean_r - self._negative_slope * mean_n
		outputs_variance = variance_r \
						   + self._negative_slope * self._negative_slope * variance_n \
						   - 2.0 * self._negative_slope * covxy
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance


class Dropout(tf.keras.Model):
	def __init__(self, p = 0.5, keep_variance_fn=None, inplace=False):
		super(Dropout, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.inplace = inplace
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
		self.p = p

	def call(self, inputs_mean, inputs_variance):
		training = tf.keras.backend.learning_phase()
		sess2 = tf.Session()
		drop_layer = tf.keras.layers.SpatialDropout2D(data_format='channels_last', rate =self.p)
		if sess2.run(training):
			binary_mask = tf.ones_like(inputs_mean)
			binary_mask = drop_layer(binary_mask, training)
			outputs_mean = inputs_mean*binary_mask
			outputs_variance = inputs_variance*binary_mask**2
		
			if self._keep_variance_fn is not None:
				outputs_variance = self._keep_variance_fn(outputs_variance)
			return outputs_mean, outputs_variance, binary_mask
		
		outputs_variance = inputs_variance
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return inputs_mean, outputs_variance

class Conv2d_(tf.keras.Model):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding='SAME', dilation=1, groups=1, bias=True,
				 keep_variance_fn=None, name_='conv'):
		super(Conv2d_, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.kernel_size = kernel_size
		self.stride = [1,stride, stride,1]
		self.padding = padding
		self.dilation = dilation
		self.name_ = name_
		self.weight_shape = [self.kernel_size, self.kernel_size, in_channels, out_channels]
		# print(self.weights)
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", 
			dtype=tf.float64, 
			shape=list(self.weight_shape))
		self.biases = tf.Variable(np.zeros(out_channels), dtype=tf.float64)

	def call(self, inputs_mean, inputs_variance):
		## For mean
		outputs_mean = tf.nn.conv2d(input= inputs_mean, filter = self.weights_, strides= self.stride,
		 padding= self.padding, name = self.name_)
		outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		## For variance
		outputs_variance = tf.nn.conv2d(input= inputs_variance, filter = self.weights_**2, strides= self.stride,
		 padding= self.padding, name = self.name_)
		outputs_variance = tf.nn.bias_add(outputs_variance, self.biases)
		
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance



# class Conv2d(_ConvNd):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True,
#                  keep_variance_fn=None, padding_mode='zeros'):
#         self._keep_variance_fn = keep_variance_fn
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         super(Conv2d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _pair(0), groups, bias, padding_mode)

#     def forward(self, inputs_mean, inputs_variance):
#         outputs_mean = F.conv2d(
#             inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         outputs_variance = F.conv2d(
#             inputs_variance, self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
#         if self._keep_variance_fn is not None:
#             outputs_variance = self._keep_variance_fn(outputs_variance)
#         return outputs_mean, outputs_variance
