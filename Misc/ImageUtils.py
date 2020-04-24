import time
import cv2
import sys
sys.dont_write_bytecode = True


def StandardizeInputs(image):
    return image

def flipImage(image):
	return cv2.flip(image,1)