import tensorflow as tf
import numpy as np
from numbers import Number
from maths import normpdf, normcdf
# from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.convolutional import Conv
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

def kaiming_normal(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)
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

	def call(self, input_mean, inputs_variance, pool_size=(2,2)):
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
	def __init__(self, keep_variance_fn=None, dtype = tf.float32):
		super(ReLU, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.dtype_ = dtype

	def call(self, feature_mean, feature_variance):
		feature_stddev = tf.cast(tf.sqrt(feature_variance), self.dtype_)
		div = feature_mean / feature_stddev
		pdf = normpdf(div)
		cdf = normcdf(div)
		output_mean = feature_mean * cdf + feature_stddev * pdf
		output_variance = (feature_mean ** 2 + feature_variance) * cdf \
						   + feature_mean * feature_stddev * pdf - output_mean ** 2
		# output_mean = tf.nn.relu(feature_mean)
		# output_variance = tf.where(output_mean==0, tf.zeros_like(feature_variance), feature_variance)
		if self._keep_variance_fn is not None:
			output_variance = self._keep_variance_fn(output_variance)
		return output_mean, output_variance

class LeakyReLU(tf.keras.Model):
	def __init__(self, negative_slope=0.01, keep_variance_fn=None, dtype = tf.float32):
		super(LeakyReLU, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self._negative_slope = negative_slope
		self.dtype = dtpe

	def call(self, features_mean, features_variance):
		features_stddev = tf.cast(tf.sqrt(features_variance), self.dtype)
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
	def __init__(self, p = 0.5, keep_variance_fn=None, inplace=False, training=False):
		super(Dropout, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.inplace = inplace
		self.isTraining = training
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
		self.p = p

	def call(self, inputs_mean, inputs_variance):
		drop_layer = tf.keras.layers.SpatialDropout2D(data_format='channels_last', rate =self.p)
		if self.isTraining:
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
		self.out_channels_ = out_channels
		self.stride = [1,stride, stride,1]
		self.strides = stride
		self.padding = padding
		self.dilation = dilation
		self.name_ = name_
		self.weight_shape = [self.kernel_size, self.kernel_size, in_channels, out_channels]
		self.dtype_ = dtype_
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", dtype = self.dtype_, shape=list(self.weight_shape))
		self.biasStatus = bias
		if self.biasStatus:
			self.biases = tf.Variable(np.zeros(out_channels), dtype = self.dtype_)		

		# self.weights_ = tf.get_variable(name=self.name_+"_Weight",
		# 	initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=self.dtype_),
		# 	shape=list(self.weight_shape), trainable=True)
		# self.weights_ = tf.Variable(kaiming_normal(self.weight_shape), name=self.name_+"_Weight", trainable=True)

	def call(self, inputs_mean, inputs_variance):
		## For mean
		# outputs_mean = tf.nn.conv2d(input= inputs_mean, filter = self.weights_, strides= self.stride,
		#  padding= self.padding, name = self.name_)
		input_shape = inputs_mean.shape.as_list()
		convLayer = Conv(rank=2, filters=self.out_channels_, input_shape=input_shape[1:4], kernel_size=self.kernel_size, 
			strides=self.strides, padding='same', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=self.dtype_),
			kernel_constraint=tf.keras.constraints.max_norm(1, [0, 1, 2]), name= self.name_)

		outputs_mean = convLayer(inputs_mean)


		if self.biasStatus:
			outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		## For variance
		outputs_variance = tf.nn.conv2d(input= inputs_variance, filter = convLayer.kernel**2, strides= self.stride,
		 padding= self.padding, name = self.name_)
		
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
	def __init__(self, inShape, outShape, bias=True, keep_variance_fn=None, 
		name_='linear', dtype_=tf.float32):
		super(Linear, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.biasStatus = bias
		self.name_ = name_
		self.dtype_ = dtype_
		self.weights_ = tf.get_variable(name=self.name_+"_Weight",dtype=self.dtype_, 
			shape=[outShape, inShape])

		# self.weights_ = tf.get_variable(name=self.name_+"_Weight", 
		# 	initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=self.dtype_),
		# 	shape=[outShape, inShape], trainable=True)

		# self.weights_ = tf.Variable(kaiming_normal([outShape, inShape]), name=self.name_+"_Weight", trainable=True)
		if self.biasStatus:
			self.biases = tf.Variable(np.zeros(outShape), dtype=self.dtype_)

	def call(self, inputs_mean, inputs_variance):
		# self.out_features = out_features
		input_shape = inputs_mean.shape.as_list()
		if len(input_shape)==4:
			input_shape = [input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]]
		# inputs_mean = tf.reshape(inputs_mean, [input_shape[0], input_shape[1]])
		# inputs_variance = tf.reshape(inputs_variance, [input_shape[0], input_shape[1]])
		outputs_mean = tf.matmul(inputs_mean, self.weights_, transpose_b=True)
		if self.biasStatus:
			outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		outputs_variance = tf.matmul(inputs_variance, self.weights_**2, transpose_b=True)
		if self._keep_variance_fn is not None:
			outputs_variance = self._keep_variance_fn(outputs_variance)
		return outputs_mean, outputs_variance


class BatchNorm2d(tf.keras.Model):
	def __init__(self, eps=1e-5, momentum=0.1, affine=True,
				 track_running_stats=True, keep_variance_fn=None, 
				 name_= 'BatchNorm2d', dtype_=tf.float32, training=False):
		super(BatchNorm2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		self.name_ = name_
		self.dtype_ = dtype_
		self.isTraining = training
		self.decay = 0.99

	def call(self, inputs_mean, inputs_variance):
		shape = inputs_mean.get_shape().as_list()
		# gamma: a trainable scale factor
		gamma = tf.get_variable(self.name_+"_gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
		
		# beta: a trainable shift value
		beta = tf.get_variable(self.name_+"_beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
		moving_avg = tf.get_variable(self.name_+"_moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
		moving_var = tf.get_variable(self.name_+"_moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
		
		beta_var = tf.constant(np.zeros(shape[-1]), dtype = self.dtype_)
		moving_avg_var = tf.constant(np.zeros(shape[-1]), dtype = self.dtype_)
		moving_var_var = tf.constant(np.ones(shape[-1]), dtype = self.dtype_)
		control_inputs = []
		if self.isTraining:
			# tf.nn.moments == Calculate the mean and the variance of the tensor x
			avg, var = tf.nn.moments(inputs_mean, range(len(shape)-1))
			update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, self.decay)
			update_moving_var = moving_averages.assign_moving_average(moving_var, var, self.decay)
			control_inputs = [update_moving_avg, update_moving_var]
		else:
			avg = moving_avg
			var = moving_var
		with tf.control_dependencies(control_inputs):
			outputs_mean = tf.nn.batch_normalization(inputs_mean, avg, var, offset=beta, scale=gamma, variance_epsilon=self.eps)
			# varGamma = tf.expand_dims(gamma,0)
			outputs_variance = tf.nn.batch_normalization(inputs_variance, moving_avg_var,
				moving_var_var, beta_var, scale=gamma**2, variance_epsilon=self.eps)
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

############################################################
#############################################################

