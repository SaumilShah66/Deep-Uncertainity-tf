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
   

def isTraining():
	training = tf.keras.backend.learning_phase()
	sess2 = tf.Session()
	return sess2.run(training)

############################################################
#############################################################

class AvgPool2d(tf.keras.Model):
	def __init__(self, pool_size=(2,2)):
		super(AvgPool2d, self).__init__()
		self.pool_size = pool_size

	def call(self, input_mean):
		pool_layer = tf.keras.layers.AvgPool2D(pool_size=self.pool_size)
		output_mean = pool_layer(input_mean)
		return output_mean

class MaxPool2d(tf.keras.Model):
	def __init__(self, pool_size=(2,2), keep_variance_fn=None):
		super(MaxPool2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.pool_size = pool_size

	def call(self, input_mean):
		pool_layer = tf.keras.layers.MaxPool2D(pool_size=self.pool_size)
		output_mean = pool_layer(input_mean)
		return output_mean

class Conv2d(tf.keras.Model):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding='SAME', dilation=1, groups=1, bias=True,
				 keep_variance_fn=None, name_='conv', dtype_ = tf.float32):
		super(Conv2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.kernel_size = kernel_size
		self.stride = [1,stride, stride,1]
		self.padding = padding
		self.dilation = dilation
		self.name_ = name_
		self.biasStatus = bias
		self.weight_shape = [self.kernel_size, self.kernel_size, in_channels, out_channels]
		self.dtype_ = dtype_
		self.weights_ = tf.get_variable(name=self.name_+"_Weight", dtype = self.dtype_, shape=list(self.weight_shape))
		if self.biasStatus:
			self.biases = tf.Variable(np.zeros(out_channels), dtype = self.dtype_)		

	def call(self, inputs_mean):
		## For mean
		outputs_mean = tf.nn.conv2d(input= inputs_mean, filter = self.weights_, strides= self.stride,
		 padding= self.padding, name = self.name_)
		if self.biasStatus:
			outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		return outputs_mean

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
		if self.biasStatus:
			self.biases = tf.Variable(np.zeros(outShape), dtype=self.dtype_)

	def call(self, inputs_mean):
		input_shape = inputs_mean.shape.as_list()
		if len(input_shape)==4:
			input_shape = [input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]]
		self.in_features = input_shape[1]
		# inputs_mean = tf.reshape(inputs_mean, [input_shape[0], input_shape[1]])
		outputs_mean = tf.matmul(inputs_mean, self.weights_, transpose_b=True)
		if self.biasStatus:
			outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		return outputs_mean

class Softmax(tf.keras.Model):
	def __init__(self, axis=0, keep_variance_fn=None):
		super(Softmax, self).__init__()
		self.axis = axis
		self._keep_variance_fn = keep_variance_fn

	def call(self, features_mean, eps=1e-5):
		log_gaussian_mean = features_mean 
		log_gaussian_mean = tf.exp(log_gaussian_mean)
		constant = tf.reduce_sum(log_gaussian_mean, axis=self.axis) + eps
		constant = tf.expand_dims(constant, axis=self.axis)
		outputs_mean = log_gaussian_mean/constant
		return outputs_mean

class ReLU(tf.keras.Model):
	def __init__(self, dtype_ = tf.float32):
		super(ReLU, self).__init__()
		self.dtype_ = dtype_

	def call(self, feature_mean):
		return tf.nn.relu(feature_mean)

class LeakyReLU(tf.keras.Model):
	def __init__(self, alpha=0.2, dtype_ = tf.float32, name = None):
		super(LeakyReLU, self).__init__()
		self.dtype_ = dtpe_
		self.alpha = alpha
		self.name = name

	def call(self, features_mean):
		return tf.nn.leaky_relu(features_mean, alpha = self.alpha, name=self.name)

class Dropout(tf.keras.Model):
	def __init__(self, p = 0.5, inplace=False, training=True):
		super(Dropout, self).__init__()
		self.inplace = inplace
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
		self.p = p

	def call(self, inputs_mean):
		drop_layer = tf.keras.layers.SpatialDropout2D(data_format='channels_last', rate =self.p)
		if isTraining():
			binary_mask = tf.ones_like(inputs_mean)
			binary_mask = drop_layer(binary_mask, training = True)
			outputs_mean = inputs_mean*binary_mask
			return outputs_mean
		else:
			return inputs_mean

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

	def call(self, inputs_mean, output_size=None):
		input_shape = inputs_mean.shape.as_list()
		outputShape = outShape(self.stride, self.kernel_size, self.padding, [input_shape[1],input_shape[2]], self.output_padding)
		self.outputShape = [input_shape[0], outputShape[0], outputShape[1], self.out_channels]
		## For mean
		outputs_mean = tf.nn.conv2d_transpose(value= inputs_mean, filter = self.weights_,output_shape=self.outputShape, strides= self.stride,
		 padding= self.padding, name = self.name_)
		outputs_mean = tf.nn.bias_add(outputs_mean, self.biases)
		return outputs_mean

class BatchNorm2d(tf.keras.Model):
	def __init__(self, eps=1e-5, momentum=0.1, affine=True,
				 track_running_stats=True, keep_variance_fn=None, 
				 name_= 'BatchNorm2d', dtype_=tf.float32):
		super(BatchNorm2d, self).__init__()
		self._keep_variance_fn = keep_variance_fn
		self.eps = 0
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		self.name_ = name_
		self.dtype_ = dtype_
		
	def call(self, inputs_mean):
		if self.track_running_stats:
			self.num_batches_tracked = tf.Variable(0, dtype=self.dtype_)
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

		outputs_mean = batchNormLayer(inputs_mean, training= (isTraining() or self.track_running_stats))
		return outputs_mean
