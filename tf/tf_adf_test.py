import tensorflow as tf 
from trials import *
import numpy as np 

def avgPool2dTest(mean, variance, pool_size):
	AvgPool2d_ = AvgPool2d()
	l = AvgPool2d_(mean, variance, pool_size)
	with tf.Session() as sess:
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	pass

def maxPool2dTest(mean, variance):
	maxPool2d = MaxPool2d()
	l = maxPool2d(mean, variance)
	with tf.Session() as sess:
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	pass

def ReLUTest(mean, variance):
	relu = ReLU()
	l = relu(mean, variance)
	with tf.Session() as sess:
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	pass

def LeakyReLUTest(mean, variance):
	relu = LeakyReLU()
	l = relu(mean, variance)
	with tf.Session() as sess:
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	pass

def DropoutTest(mean, variance):
	drop = Dropout()
	l = drop(mean, variance)
	with tf.Session() as sess:
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	pass

def Conv2dTest(mean, variance):
	conv = Conv2d_(in_channels=1, out_channels=1, kernel_size=3)
	l = conv(mean, variance)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	pass	

mean_ = np.array([[-1,  2,  3,  4], [5,  6,  7,  8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float64)
mean_ = np.reshape(mean_, (-1,4,4,1))
mean = tf.convert_to_tensor(mean_)
variance = tf.zeros_like(mean) + 0.001
pool_size = (2, 2)

# avgPool2dTest(mean, variance, pool_size)
# DropoutTest(mean, variance)
# ReLUTest(mean, variance)
# LeakyReLU(mean, variance)
Conv2dTest(mean, variance)



