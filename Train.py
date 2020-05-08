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
# termcolor, do (pip install termcolor)

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
# from Network.NetworkADF import CIFAR_ADF
# from Network.NetworkADF_NOBN import CIFAR_ADF
# from Network.RESnet import CIFAR10Model
# from Network.DenseNet import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
try:
	from StringIO import StringIO ## for Python 2
except ImportError:
	from io import StringIO ## for Python 3
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
# from Network.Layers import isTraining
# Don't generate pyc codes
sys.dont_write_bytecode = True
from tensorflow.keras import backend as K
	
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, method):
	"""
	Inputs: 
	BasePath - Path to CIFAR10 folder without "/" at the end
	DirNamesTrain - Variable with Subfolder paths to train files
	NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
	TrainLabels - Labels corresponding to Train
	NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
	ImageSize - Size of the Image
	MiniBatchSize is the size of the MiniBatch
	Outputs:
	I1Batch - Batch of images
	LabelBatch - Batch of one-hot encoded labels 
	"""
	I1Batch = []
	LabelBatch = []
	Variances = []
	ImageNum = 0
	while ImageNum < MiniBatchSize:
		# Generate random image
		RandIdx = random.randint(0, len(DirNamesTrain)-1)
		
		RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.png'  
		ImageNum += 1
		
		img = np.float32(cv2.imread(RandImageName))

		I1 = iu.StandardizeInputs(img, method=method)
		
		I1Batch.append(I1)
		Variances.append(np.zeros_like(I1)+0.001)
		Label = convertToOneHot(TrainLabels[RandIdx], 10)
		LabelBatch.append(Label)
	# 	print("Randoms --- ", RandIdx, "--",TrainLabels[RandIdx])
	# print("Labels original ---",LabelBatch)
	return I1Batch, Variances, LabelBatch



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
	"""
	Prints all stats with all arguments
	"""
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Factor of reduction in training data is ' + str(DivTrain))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	if LatestFile is not None:
		print('Loading latest checkpoint with the name ' + LatestFile)              

	

def TrainOperation(ImgPH, VarPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, lr, DirNamesValid, ValidLabels, NumValidSamples, method):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to CIFAR10 folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""      
	# Predict output with forward pass
	# cifar = CIFAR_ADF()
	cifar = CIFARNormal(training=True)
	prLogits, prSoftMax = cifar.network(ImgPH)
	# prLogits, prSoftMax = cifar.network(ImgPH, VarPH)

	with tf.name_scope('TrainingLoss'):
		train_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = LabelPH, logits = prLogits)
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  		# reg_constant = 0.001
		TrainLoss = tf.reduce_mean(train_cross_entropy) + tf.reduce_sum(reg_losses)
		trainLossSummary = tf.summary.scalar("TrainingLossEveryIter", TrainLoss)

	with tf.name_scope('TrainAccuracy'):
		prSoftMaxDecodedT = tf.argmax(prSoftMax, axis=1)
		LabelDecodedT = tf.argmax(LabelPH, axis=1)
		TrainAcc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecodedT, LabelDecodedT), dtype=tf.float32))
		trainingAccuracySummary = tf.summary.scalar("TrainingAccuracy", TrainAcc)
	
	with tf.name_scope('Adam'):
		Optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(TrainLoss)

	#### Training summary
	TrainingSummary = tf.summary.merge([trainLossSummary, trainingAccuracySummary])
	#### Validation summary
	# ValidationSummary = tf.summary.merge([validationLossSummary, validationAccuracySummary])
	
	# Setup Saver
	Saver = tf.train.Saver()
	for i, var in enumerate(Saver._var_list):
		print('Var {}: {}'.format(i, var))
	# break
	TotalAcc = []
	temp_acc = []
	temp_loss = []
	TotalLoss = []
	with tf.Session() as sess:       
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			# Extract only numbers from the name
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		# Tensorboard
		Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
			
		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				### Batch generation
				I1Batch, VarBatch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, method)
				FeedDict = {ImgPH: I1Batch, VarPH: VarBatch, LabelPH: LabelBatch}
				_, LossThisBatch, TSummary, TAcc, deco, soft, lab = sess.run([Optimizer, TrainLoss, TrainingSummary, TrainAcc, prSoftMaxDecodedT, prSoftMax, LabelDecodedT], feed_dict=FeedDict)
				# print("For debug Training "+"----"*10)
				# print("Softmax --- ",soft)
				# print("Decoded -- ", deco)
				# print("Labels --- ",lab)
				# print("Accuracy -- ", TAcc)
				# print("----"*10)
				temp_loss.append(LossThisBatch)
				TotalLoss.append(LossThisBatch)
				temp_acc.append(TAcc)
				TotalAcc.append(TAcc)
				# Save checkpoint every some SaveCheckPoint's iterations
				if PerEpochCounter % 10 == 0:
					# Save the Model learnt in this epoch
					print("Accuracy of model : " + str(sum(temp_acc)/len(temp_acc)))
					print("Loss of model : "+str(sum(temp_loss)))
					temp_loss, temp_acc = [], []

				# Tensorboard
				Writer.add_summary(TSummary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				# If you don't flush the tensorboard doesn't update until a lot of iterations!
				Writer.flush()

			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			print('\n' + SaveName + ' Model Saved... ')
			print("--"*10+"After Epoch"+"--"*10)
			print("Total Test Loss = "+str(sum(TotalLoss)))
			print("Total Test Accuracy = "+str(sum(TotalAcc)/len(TotalAcc)))
			print("--"*20)
			TotalAcc = []
			TotalLoss = []

def main():
	"""
	Inputs: 
	None
	Outputs:
	Runs the Training and testing code based on the Flag
	"""
	# Parse Command Line arguments

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='../CIFAR10', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/CIFAR10')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--NumEpochs', type=int, default=30, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='../Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
	Parser.add_argument('--Lr', type=float, default=0.01, help='Learning rate')
	Parser.add_argument('--meth', type=int, default=0, help='image std method')
	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	lr = Args.Lr
	# Setup all needed parameters including file reading
	print("Going to read files")
	DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
	DirNamesValid, NumValidSamples, ValidLabels = setupValidation(BasePath)
	print("Parameters has been set properly...")


	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None
	
	# Pretty print stats
	PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

	# Define PlaceHolder variables for Input and Predicted output
	ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
	VarPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
	LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
	
	TrainOperation(ImgPH, VarPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, lr, DirNamesValid, ValidLabels, NumValidSamples, Args.meth)
		
	
if __name__ == '__main__':
	main()
 
