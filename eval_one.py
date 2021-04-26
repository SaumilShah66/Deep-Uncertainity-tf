#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import sys
try:
	import cv2
except:
	sys.path.remove(sys.path[2])
	import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
import Misc.ImageUtils as iu
import random
# from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.NetworkADF import CIFAR_ADF
# from Network.RESnet import CIFAR10Model
# from Network.DenseNet import CIFAR10Model
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')
    return ImageSize, DataPath
    
def ReadImages(ImageSize, ImageName, method):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    I1S = iu.StandardizeInputs(np.float32(I1), randomFlip=False, method=method)

    I1Combined = np.expand_dims(I1S, axis=0)
    Varience = np.zeros_like(I1Combined) + 0.001
    return I1Combined, Varience, I1
                

def TestOperation(ImgPH, VarPH, ImageSize, ModelPath, name, method):
    Length = ImageSize[0]
    cifar = CIFAR_ADF()
    # prLogits, prSoftMaxS, Variances = cifar.network(ImgPH, VarPH)
    prLogits, prSoftMaxS = cifar.network(ImgPH, VarPH)
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        Img, Var, ImgOrg = ReadImages(ImageSize, name, method)
        FeedDict = {ImgPH: Img, VarPH: Var}
        
        # PredT, var = sess.run([prSoftMaxS, Variances], FeedDict)
        PredT= sess.run(prSoftMaxS, FeedDict)
        #PredT = np.argmax(PredT)
            #var = var.ravel()[PredT]
        # print("With mean -- ", PredT[0])
        # print()
        # print("With Variance -- ",PredT[1])
        PredT_class = np.argmax(PredT[0])
        print("Prediction is -- ",PredT_class)
        print("Prediction uncertainty -- ",PredT[1][0][PredT_class])
            
def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--Epochs', dest='Epochs', default=0, help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--ImagePath', default="../CIFAR10/Test/")
    Parser.add_argument('--Name', default="54.png")
    Parser.add_argument('--meth', type=int, default=0, help='image std method')
    
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    epoch = Args.Epochs
    name = Args.ImagePath + Args.Name
    ImageSize = [32,32,3]    
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    model_path = ModelPath+str(epoch)+"model.ckpt"
    
    LabelsPath = Args.LabelsPath
    tf.reset_default_graph()


    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    VarPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
    TestOperation(ImgPH, VarPH, ImageSize, model_path, name, Args.meth)    
     
if __name__ == '__main__':
    main()
 
