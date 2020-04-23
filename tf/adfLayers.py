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
	def __init__(self, keep_variance_fn=None, dtype_ = tf.float32):
		super(ReLU, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.dtype_ = dtype_

	def call(self, feature_mean, feature_variance):
		feature_stddev = tf.cast(tf.sqrt(feature_variance), self.dtype_)
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
	def __init__(self, negative_slope=0.01, keep_variance_fn=None, dtype_ = tf.float32):
		super(LeakyReLU, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self._negative_slope = negative_slope
		self.dtype_ = dtpe_

	def call(self, features_mean, features_variance):
		features_stddev = tf.cast(tf.sqrt(features_variance), self.dtype_)
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
		drop_layer = tf.keras.layers.SpatialDropout2D(data_format='channels_last', rate =self.p)
		if isTraining():
			binary_mask = tf.ones_like(inputs_mean)
			binary_mask = drop_layer(binary_mask, training = True)
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
				 keep_variance_fn=None, name_='conv', dtype_ = tf.float32):
		super(Conv2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.kernel_size = kernel_size
		self.stride = [1,stride, stride,1]
		self.out_channels = out_channels
		# self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.name_ = name_
		self.weight_shape = [self.kernel_size, self.kernel_size, in_channels, out_channels]
		self.dtype_ = dtype_
		self.weights_ = tf.get_variable(name=self.name+"_Weight", trainable=True ,dtype = self.dtype_, shape=list(self.weight_shape))
		# self.weights_ = tf.Variable(np.ones(list(self.weight_shape)) ,dtype = self.dtype_)
		self.biases = tf.Variable(np.zeros(out_channels), dtype = self.dtype_)		

	def call(self, inputs_mean, inputs_variance):
		## For mean
		outputs_mean = tf.nn.conv2d(input= inputs_mean, filter = self.weights_, strides= self.stride,
		 padding= self.padding, name = self.name_)
		outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		## For variance
		outputs_variance = tf.nn.conv2d(input= inputs_variance, filter = (self.weights_*self.weights_), strides= self.stride,
		 padding= self.padding, name = self.name_)
		
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance
		

	# def call(self, inputs_mean, inputs_variance):
	# 	## For mean
	# 	# rslt = tf.layers.conv2d(inputs = rslt, name='conv1', padding='same',filters = 16, kernel_size = 3, activation = None)


	# 	# wt_initializer = tf.constant_initializer(WEIGHT_MATRIX)
	# 	# wtv_initializer = tf.constant_initializer(WEIGHT_MATRIX**2)
	# 	# bs_initializer = tf.constant_initializer(self.biases)

	# 	conv2d_layer= tf.keras.layers.Convolution2D(filters= self.out_channels, kernel_size= self.kernel_size,
	# 												strides= self.stride, padding= self.padding,dilation_rate= self.dilation,
	# 												use_bias=False,kernel_initializer= None, bias_initializer= None)

	# 	outputs_mean = conv2d_layer(inputs_mean)
	# 	## For variance
	# 	conv2d_layer= tf.keras.layers.Convolution2D(filters= self.out_channels, kernel_size= self.kernel_size,
	# 												strides= self.stride, padding= self.padding,dilation_rate= self.dilation,
	# 												use_bias=True,kernel_initializer= None, bias_initializer= None)


	# 	outputs_variance = conv2d_layer(inputs_variance)		
	# 	if self._keep_variance_fn is not None:
	# 		outputs_variance = self._keep_variance_fn(outputs_variance)
	# 	return outputs_mean, outputs_variance

		

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
	# strideList is of the form [_,h,w,_]
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
				 keep_variance_fn=None, name_='convTrans', output_size=None, dtype=tf.float32):
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
		self.dtype = dtype
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", dtype=self.dtype, shape=list(self.weight_shape))
		self.biases = tf.Variable(np.zeros(out_channels), dtype=self.dtype)

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
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance


class Linear(tf.keras.Model):
	def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None, name_='linear', dtype_=tf.float32):
		super(Linear, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.biasStatus = bias
		self.name_ = name_
		self.dtype_ = dtype_
		self.in_features = in_features
		self.out_features = out_features
		weight_shape = [self.out_features, self.in_features]
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", dtype=self.dtype_, shape=list(weight_shape))

	def call(self, inputs_mean, inputs_variance):
		input_shape = inputs_mean.shape.as_list()
		inputs_mean = tf.reshape(inputs_mean, [input_shape[0], self.in_features])
		inputs_variance = tf.reshape(inputs_variance, [input_shape[0], self.in_features])
		outputs_mean = tf.matmul(inputs_mean, self.weights_, transpose_b=True)
		if self.biasStatus:
			self.biases = tf.Variable(np.zeros(self.out_features), dtype=self.dtype_)
			outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		outputs_variance = tf.matmul(inputs_variance, self.weights_**2, transpose_b=True)
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance


class BatchNorm2d(tf.keras.Model):
	def __init__(self, eps=1e-5, momentum=0.1, affine=True,
				 track_running_stats=True, keep_variance_fn=None, 
				 name_= 'BatchNorm2d', dtype_=tf.float32):
		super(BatchNorm2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		self.name_ = name_
		self.dtype_ = dtype_
		
	def call(self, inputs_mean, inputs_variance):
		# exponential_average_factor is self.momentum set to
		# (when it is available) only so that if gets updated
		# in ONNX graph when this node is exported to ONNX.
		if self.track_running_stats:
			self.num_batches_tracked = tf.Variable(0, dtype=tf.int64)

		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum
		if isTraining() and self.track_running_stats:
			if self.num_batches_tracked is not None:
				self.num_batches_tracked.assign_add(1)
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / self.num_batches_tracked
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		input_shape = inputs_mean.shape.as_list()
		weights = tf.random_uniform(input_shape[-1:], minval=0.0, maxval=1.0)
		
		bias = tf.zeros(input_shape[-1], dtype=self.dtype_)

		batchNormLayer = tf.keras.layers.BatchNormalization(momentum= exponential_average_factor, epsilon=self.eps,
															moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer(),
															center=False, scale=False, trainable=True,
															adjustment= lambda input_shape: ( weights, bias))

		outputs_mean = batchNormLayer(tf.cast(inputs_mean, dtype=self.dtype_), training= (isTraining() or self.track_running_stats))
		outputs_variance = inputs_variance
		outputs_variance = outputs_variance * (tf.cast(weights, dtype=self.dtype_)**2)
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance

class Softmax(tf.keras.Model):
	def __init__(self, axis=0, keep_variance_fn=None):
		super(Softmax, self).__init__()
		self.axis = axis
		self._keep_variance_fn = keep_variance_fn

	def call(self, features_mean, features_variance, eps=1e-5):
		"""Softmax function applied to a multivariate Gaussian distribution.
		It works under the assumption that features_mean and features_variance 
		are the parameters of a the indepent gaussians that contribute to the 
		multivariate gaussian. 
		Mean and variance of the log-normal distribution are computed following
		https://en.wikipedia.org/wiki/Log-normal_distribution."""
		  		
		log_gaussian_mean = features_mean + 0.5 * features_variance
		log_gaussian_variance = 2 * log_gaussian_mean

		log_gaussian_mean = tf.exp(log_gaussian_mean)
		log_gaussian_variance = tf.exp(log_gaussian_variance)
		log_gaussian_variance = log_gaussian_variance*(tf.exp(features_variance)-1)

		constant = tf.reduce_sum(log_gaussian_mean, axis=self.axis) + eps
		constant = tf.expand_dims(constant, axis=self.axis)

		outputs_mean = log_gaussian_mean/constant
		outputs_variance = log_gaussian_variance/(constant**2)
		
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance

def isTraining():
	training = tf.keras.backend.learning_phase()
	sess2 = tf.Session()
	return sess2.run(training)
