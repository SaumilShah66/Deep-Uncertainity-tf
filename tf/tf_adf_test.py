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
	print('zmean')
	print(zmean)
	pass

mean_ = np.array([[1,  2,  3,  4], [5,  6,  7,  8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float64)
mean_ = np.reshape(mean_, (1,4,4,1))
mean = tf.convert_to_tensor(mean_)
variance = tf.zeros_like(mean) + 0.001
pool_size = (2, 2)
# avgPool2dTest(mean, variance, pool_size)
maxPool2dTest(mean, variance)
