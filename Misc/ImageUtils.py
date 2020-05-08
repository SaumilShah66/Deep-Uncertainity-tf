import time
import cv2
import sys
import random
import numpy as np
sys.dont_write_bytecode = True


def StandardizeInputs(image, randomFlip=True, method=0):
	if(method == 0):
		image = meth0(image)
	elif(method == 1):
		image = meth1(image)
	elif(method == 2):
		image = meth2(image)
	elif(method == 3):
		image = meth3(image)
	elif(method == 4):
		image = meth4(image)
	elif(method == 5):
		image = meth5(image)

	if randomFlip:
		return flipImage(image)
	else:
		return image

def flipImage(image):
	if random.randint(0,1):
		return cv2.flip(image,1)
	else:
		return image

def meth0(image):
	image = image/255.0
	NormalizeMean = np.array([0.4914,0.4822,0.4465])
	NormalizeStd = np.array([0.2023,0.1994,0.2010])
	for i in range(3):
		image[:,:,i] = (image[:,:,i] - NormalizeMean[i])/NormalizeStd[i]
	return image

def meth1(image): # scaled ndividually in range -1 to 1
	image = image/255.0
	for i in range(3):
		mean = np.mean(image[:,:,i])
		std = np.std(image[:,:,i])
		image[:,:,i] = (image[:,:,i] - mean)/std
	return image

def meth2(image): # in range -1 to 1
	NormalizeMean = np.array([127.0,127.0,127.0])
	NormalizeStd = np.array([255.0,255.0,255.0])
	for i in range(3):
		image[:,:,i] = (image[:,:,i] - NormalizeMean[i])/NormalizeStd[i]
	return image

def meth3(image): # in range 0 to 1
	NormalizeMean = np.array([0,0,0])
	NormalizeStd = np.array([255.0,255.0,255.0])
	for i in range(3):
		image[:,:,i] = (image[:,:,i] - NormalizeMean[i])/NormalizeStd[i]
	return image

def meth4(image): # in range -0.5 to 0.5
	NormalizeMean = np.array([127,127,127])
	NormalizeStd = np.array([255.0,255.0,255.0])
	for i in range(3):
		image[:,:,i] = (image[:,:,i] - NormalizeMean[i])/(2*NormalizeStd[i])
	return image

def meth5(image): # in range 0 to 0.5
	NormalizeMean = np.array([0,0,0])
	NormalizeStd = np.array([255.0,255.0,255.0])
	for i in range(3):
		image[:,:,i] = (image[:,:,i] - NormalizeMean[i])/(2*NormalizeStd[i])
	return image