import tensorflow as tf
import numpy as np
from numbers import Number
from maths import normpdf, normcdf
# from tensorflow.python.keras import backend as K
from tensorflow.python.training import moving_averages
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
	status = sess2.run(training)
	sess2.close()
	print(status)
	return status

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
		self.isTraining = training
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
		self.p = p

	def call(self, inputs_mean):
		drop_layer = tf.keras.layers.SpatialDropout2D(data_format='channels_last', rate =self.p)
		if self.isTraining:
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
				 name_= 'BatchNorm2d', dtype_=tf.float32, training=True):
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

	def call(self, inputs_mean):
		# with tf.variable_scope('BatchNorm_'+self.name_, reuse=True) as bnscope:
		shape = inputs_mean.get_shape().as_list()
		# gamma: a trainable scale factor
		gamma = tf.get_variable(self.name_+"_gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=self.isTraining)
		# beta: a trainable shift value
		beta = tf.get_variable(self.name_+"_beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=self.isTraining)
		moving_avg = tf.get_variable(self.name_+"_moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
		moving_var = tf.get_variable(self.name_+"_moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
		if self.isTraining:
			# tf.nn.moments == Calculate the mean and the variance of the tensor x
			avg, var = tf.nn.moments(inputs_mean, range(len(shape)-1))
			update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, self.decay)
			update_moving_var = moving_averages.assign_moving_average(moving_var, var, self.decay)
			control_inputs = [update_moving_avg, update_moving_var]
		else:
			avg = moving_avg
			var = moving_var
			control_inputs = []
		with tf.control_dependencies(control_inputs):
			output = tf.nn.batch_normalization(inputs_mean, avg, var, offset=beta, scale=gamma, variance_epsilon=self.eps)
		return output
