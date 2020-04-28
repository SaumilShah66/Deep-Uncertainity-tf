import time
import cv2
import sys
import random
import numpy as np
sys.dont_write_bytecode = True


def StandardizeInputs(image):
	image = image/255.0
	NormalizeMean = np.array([0.4914,0.4822,0.4465])
	NormalizeStd = np.array([0.2023,0.1994,0.2010])
	for i in range(3):
		image[:,:,i] = (image[:,:,i] - NormalizeMean[i])/NormalizeStd[i]
	if random.randint(0,1):
		return flipImage(image)
	else:
		return image

def flipImage(image):
	return cv2.flip(image,1)