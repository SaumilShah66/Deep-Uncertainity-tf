import tensorflow as tf
import numpy as np
from numbers import Number
from maths import normpdf, normcdf
# from tensorflow.python.keras import backend as K

#########################
## start interpolation.py
#########################

def resize2D_as(inputs, output_as, mode="bilinear"):
	size_targets = output_as.shape.as_list()
	size_targets = [size_targets[1], size_targets[2]]
	return resize2D(inputs, size_targets, mode=mode)

def resize2D(inputs, size_targets, mode="bilinear"):
	size_inputs = inputs.shape.as_list()
	size_inputs = [size_inputs[1], size_inputs[2]]
	h1, w1 = size_inputs[0], size_inputs[1]
	h2, w2 = size_targets[0], size_targets[1]
	if all([size_inputs == size_targets]):
		return inputs  # nothing to do
	elif any([size_targets < size_inputs]):
		pool_size = (int(round(h1*1.0/h2)), int(round(w1*1.0/w2)))
		pool_layer = tf.keras.layers.AvgPool2D(pool_size=pool_size, padding='same')
		resized = pool_layer(inputs)
	else:
		pool_size = (int(round(h2 * 1.0 / h1)), int(round(w2 * 1.0 / w1)))
		print('ppol', pool_size)
		upLayer = tf.keras.layers.UpSampling2D(size=pool_size)
		resized = upLayer(inputs= inputs)
	return resized

#######################
## end interpolation.py
#######################



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

class Conv2d(tf.keras.Model):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding='SAME', dilation=1, groups=1, bias=True,
				 keep_variance_fn=None, name_='conv'):
		super(Conv2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.kernel_size = kernel_size
		self.stride = [1,stride, stride,1]
		self.padding = padding
		self.dilation = dilation
		self.name_ = name_
		self.weight_shape = [self.kernel_size, self.kernel_size, in_channels, out_channels]
		# print(self.weights)
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", dtype=tf.float64, shape=list(self.weight_shape))
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

##################################################
#### concatenate_as function not tested separately
##################################################
def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
	means = [resize2D_as(x[0], tensor_as[0], mode=mode) for x in tensor_list]
	variances = [resize2D_as(x[1], tensor_as[0], mode=mode) for x in tensor_list]
	means = tf.concat(means, dim=dim)
	variances = tf.concat(variances, dim=dim)
	return means, variances


def outShape(strideList, filter_size, padding, input_length, output_padding=0): #ref https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
	# strideList is of the form [1,h,w,1]
	length = np.ones(2, dtype =np.int32)
	if output_padding == 0:
		if padding == 'VALID' or padding == 'valid':
			# note the call to `max` below!
			length[0] = input_length[0] * strideList[1] + max(filter_size - strideList[1], 0)
			length[1] = input_length[1] * strideList[2] + max(filter_size - strideList[2], 0)
		elif padding == 'FULL' or padding == 'full':
			length[0] = input_length[0] * strideList[1] - (strideList[1] + filter_size - 2)
			length[1] = input_length[1] * strideList[2] - (strideList[2] + filter_size - 2)
		elif padding == 'SAME' or padding == 'same':
			length[0] = input_length[0] * strideList[1]
			length[1] = input_length[1] * strideList[2]
	else:
		if padding == 'SAME' or padding == 'same':
		  pad = filter_size // 2
		elif padding == 'VALID' or padding == 'valid':
		  pad = 0
		elif padding == 'FULL' or padding == 'full':
		  pad = filter_size - 1

		length[0] = ((input_length[0] - 1) * strideList[1] + filter_size - 2 * pad +
				  output_padding)
		length[1] = ((input_length[1] - 1) * strideList[2] + filter_size - 2 * pad +
				  output_padding)
	return length

class ConvTranspose2d(tf.keras.Model):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding='SAME', output_padding=0, groups=1, bias=True, dilation=1,
				 keep_variance_fn=None, name_='convTrans', output_size=None):
		super(ConvTranspose2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.kernel_size = kernel_size
		self.stride = [1,stride, stride,1]
		self.padding = padding
		self.output_padding = output_padding
		self.dilation = dilation
		self.name_ = name_
		self.out_channels = out_channels
		self.weight_shape = [self.kernel_size, self.kernel_size, out_channels, in_channels]
		# print(self.weights)
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", dtype=tf.float64, shape=list(self.weight_shape))
		self.biases = tf.Variable(np.zeros(out_channels), dtype=tf.float64)

	def call(self, inputs_mean, inputs_variance, output_size=None):
		input_shape = inputs_mean.shape.as_list()
		outputShape = outShape(self.stride, self.kernel_size, self.padding, [input_shape[1],input_shape[2]], self.output_padding)
		self.outputShape = [input_shape[0], outputShape[0], outputShape[1], self.out_channels]
		## For mean
		outputs_mean = tf.nn.conv2d_transpose(value= inputs_mean, filter = self.weights_,output_shape=self.outputShape, strides= self.stride,
		 padding= self.padding, name = self.name_)
		outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		## For variance
		outputs_variance = tf.nn.conv2d_transpose(value= inputs_variance, output_shape=self.outputShape, filter = self.weights_**2, strides= self.stride,
		 padding= self.padding, name = self.name_)
		outputs_variance = tf.nn.bias_add(outputs_variance, self.biases)
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance


class Linear(tf.keras.Model):
	def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None, name_='linear'):
		super(Linear, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.in_features = in_features
		self.out_features = out_features
		self.weight_shape = [self.out_features, self.in_features]
		self.biasStatus = bias
		self.name_ = name_
		# print(self.weights)
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", dtype=tf.float64, shape=list(self.weight_shape))
		if bias:
			self.biases = tf.Variable(np.zeros(self.out_features), dtype=tf.float64)

	def call(self, inputs_mean, inputs_variance):
		input_shape = inputs_mean.shape.as_list()
		inputs_mean = tf.reshape(inputs_mean, [input_shape[1], input_shape[2]])
		inputs_variance = tf.reshape(inputs_variance, [input_shape[1], input_shape[2]])
		outputs_mean = tf.matmul(inputs_mean, self.weights_, transpose_b=True)
		if self.biasStatus:
			outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		outputs_variance = tf.matmul(inputs_variance, self.weights_**2, transpose_b=True)
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance


# class Linear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None):
#         super(Linear, self).__init__()
#         self._keep_variance_fn = keep_variance_fn
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, inputs_mean, inputs_variance):
#         outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
#         outputs_variance = F.linear(inputs_variance, self.weight**2, None)
#         if self._keep_variance_fn is not None:
#             outputs_variance = self._keep_variance_fn(outputs_variance)
#         return outputs_mean, outputs_variance