import tensorflow as tf 
from trials import *
import numpy as np 
try:
	import cv2
except:
	import sys
	sys.path.remove(sys.path[2])
	import cv2

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
	conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
	l = conv(mean, variance)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	print(outVar.shape)
	pass	

def ConvTranspose2dTest(mean, variance):
	deconv = ConvTranspose2d(1, 2, 3, 2)
	l = deconv(mean, variance)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	print(outVar.shape)
	pass

def LinearTest(mean, variance):
	linearLayer = Linear()
	l = linearLayer(mean, variance, 10)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	print(outVar.shape)
	pass

def BatchNorm2dTest(mean, variance):
	batchNormLayer = BatchNorm2d()
	l = batchNormLayer(mean, variance)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		outMean, outVar = sess.run(l)
	print("Mean " + "-" * 20)
	print(outMean)
	print("Variance " + "-" * 20)
	print(outVar)
	print(outVar.shape)
	print(outMean.shape)
	# print(wt)
	pass


mean_ = np.array([[1,  2,  3,  4], [5,  6,  7,  8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float64)
# mean_ = np.dstack([mean_, mean_])
mean_ = np.reshape(mean_, (-1,4,4,1))
mean = tf.convert_to_tensor(mean_)
variance = tf.zeros_like(mean) + 0.001
pool_size = (2, 2)

# img = cv2.imread('0.jpg',0)
# img = img.reshape(1,img.shape[0], img.shape[1],1)
# imgt = tf.convert_to_tensor(img, dtype = tf.float64)
# new = resize2D(imgt, [200,200])
# with tf.Session() as sess:
# 	new1 = sess.run(new)
# 	print(new1.shape)
	# new2 = new1.reshape()
# cv2.imwrite('new.jpg', new1)

# avgPool2dTest(mean, variance, pool_size)
DropoutTest(mean, variance)
# ReLUTest(mean, variance)
# LeakyReLU(mean, variance)
# Conv2dTest(mean, variance)
# ConvTranspose2dTest(mean, variance)
# LinearTest(mean, variance)
# BatchNorm2dTest(mean, variance)
