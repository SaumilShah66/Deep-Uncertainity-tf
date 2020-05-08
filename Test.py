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
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import CIFARNormal
# from Network.Network import CIFARNormal
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
    # NumImages = 10
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')
    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath, method):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    I1S = iu.StandardizeInputs(np.float32(I1), randomFlip=False, method= method)

    I1Combined = np.expand_dims(I1S, axis=0)
    Varience = np.zeros_like(I1Combined) + 0.001
    return I1Combined, Varience, I1
                

def TestOperation(ImgPH, VarPH, ImageSize, ModelPath, DataPath, LabelsPathPred, method):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    # _, prSoftMaxS = CIFAR10Model(ImgPH, ImageSize, 1)
    cifar = CIFARNormal(training=False)
    prLogits, prSoftMaxS = cifar.network(ImgPH)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        # for i, var in enumerate(Saver._var_list):
        #     print('Var {}: {}'.format(i, var))
        # break
        OutSaveT = open(LabelsPathPred, 'w')

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, Var, ImgOrg = ReadImages(ImageSize, DataPathNow, method)
            FeedDict = {ImgPH: Img, VarPH: Var}
            Prediction, logits = sess.run([prSoftMaxS, prLogits], FeedDict)
            # print("----"*10)
            # print("Softmax values -- ",Prediction)
            # print("Logits ---",logits)
            PredT = np.argmax(Prediction)
            # print(PredT)
            # print("---"*10)
            # print("Training status -- ",isTraining())
            # print(str(count)+)
            OutSaveT.write(str(PredT)+'\n')
            
        OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    # print("Predictions ---\n",Pred)
    # print("Actual ----\n",GT)
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """
    LabelTest = LabelsTrue
    LabelPred = LabelsPred
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelTest,  # True class for test-set.
                          y_pred=LabelPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelPred, LabelTest)), '%')
    return Accuracy(LabelPred, LabelTest)

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--Epochs', dest='Epochs', default=19, help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--meth', type=int, default=0, help='image std method')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    epoch = Args.Epochs
    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    acc = []
    # for epoch in range(epochs):
    model_path = ModelPath+str(epoch)+"model.ckpt"
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    tf.reset_default_graph()
    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    VarPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
    TestOperation(ImgPH, VarPH, ImageSize, model_path, DataPath, LabelsPathPred, Args.meth)    
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    print(Accuracy(LabelsPred,LabelsTrue))
    acc.append(ConfusionMatrix(LabelsTrue, LabelsPred))
        
    # plt.plot(acc)
    # print(acc)
    # plt.show()
    # print(acc)

def test():
    path = "../CIFAR10/Train/19.png"
    I1S = iu.StandardizeInputs(np.float32(cv2.imread(path)))
    I1Combined = np.expand_dims(I1S, axis=0)
    ImageSize = [32, 32, 3]
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelPH = tf.placeholder(tf.float32, shape=(1, 10)) 
    cifar = CIFARNormal()
    prLogits, prSoftMaxS = cifar.network(ImgPH)

    # with tf.name_scope('ValidAccuracy'):
    #     prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
    #     LabelDecoded = tf.argmax(LabelPH, axis=1)
    #     ValidAcc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))
    
    model_path = "../Checkpoints/198model.ckpt"
    Saver = tf.train.Saver()
    for i, var in enumerate(Saver._var_list):
        print('Var {}: {}'.format(i, var))
    
    with tf.Session() as sess:
        Saver.restore(sess, model_path)
        FeedDict = {ImgPH: I1Combined}
        PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))
    print(PredT)

if __name__ == '__main__':
    main()
    # test()
 
