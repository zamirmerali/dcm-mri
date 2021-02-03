#Dicom Converter
#Copyright, Zamir Merali, April 27, 2018
#This module accomplishes the following task:
#Input: Many patient's ax_t2 images, slice labels
#The guts: Dicoms are normalized to slice thickness, a map is applied, more pre-processing. 
#Output: Each slice is loaded to a numpy array, the label is read from the csv, and a jepg is saved and named. 
#Output: Jpegs that are labeled compressed1,2,3... and notcompressed1,2,3. 

#Dependancies
import os
from os.path import join, getsize, dirname
from pprint import pprint

import numpy
import numpy as np
import csv

import pydicom
from pydicom.filereader import read_dicomdir

import matplotlib.pyplot as plt 
#from matplotlib import pyplot, cm
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from PIL.Image import fromarray


#Define list of patient ids
patient_id = []
#Define lists for filepaths
ax_t2_filepath = []
#Define the root directory
rootDir = '/Users/zamirmerali/aospine/sapien_tech/more_mris/NCR'
rater1Dir = '/Users/zamirmerali/aospine/sapien_tech/more_mris/NCR'
rater2Dir = '/Users/zamirmerali/aospine/sapien_tech/more_mris/NCR'
#Fill list of patient ids
patient_id = os.listdir(rootDir)
print patient_id
#Current patient id
ident = 2

#Function, take patient id, output list of file paths for ax t2 dicoms
def paths_ax_t2(patient_id):
	ax_t2_filepath = []
	subDir = os.path.join(rootDir,patient_id,'ax_t2','patient')
	print(subDir)
	for root, dirs, files in os.walk(subDir):
		for fname in files:
			if fname.endswith('.dcm'):
				img_path = os.path.join(root,fname)
				ax_t2_filepath.append(img_path)
	return ax_t2_filepath

for k in range(0,25):
	t2_filepath = []
	print(patient_id[k])
	if not patient_id[k] == '.DS_Store':
		t2_filepath = paths_ax_t2(patient_id[k])
	print(len(t2_filepath))

	#Get ref file
	RefDs = pydicom.read_file(t2_filepath[0])
	#print RefDs

	#Load dimensions of the dicom sequence
	ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(t2_filepath))

	#Load spacing values in mm
	#ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

	#Calculate axes for the array
	#x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
	#y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
	#z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

	# Make empty array based on extracted dimensions
	ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

	# loop through all the DICOM files
	for filenameDCM in t2_filepath:
	    # read the file
	   	ds = pydicom.read_file(filenameDCM)
	    # store the raw image data
 	   	ArrayDicom[:, :, t2_filepath.index(filenameDCM)] = ds.pixel_array 

	print(ConstPixelDims)
	#print(ConstPixelSpacing)

	plt.ion()
	#initialize a list to store the categories
	category1 = []
	category2 = []
	#load the filepath for the csv file
	csvfile1 = os.path.join(rater1Dir,patient_id[k],'compression_label.csv')
	csvfile2 = os.path.join(rater2Dir,patient_id[k],'compression_label.csv')
	with open (csvfile1, 'rU') as f:
		category1 = list(csv.reader(f))
	with open (csvfile2, 'rU') as g:
		category2 = list(csv.reader(g))


	#print category

	#test code that checks an entry in the category list
	#if category[1] == ['0']:
	#	print ("notcompressed")

	for i in range(len(t2_filepath)):
		if (category1[i] == ['0']) & (category2[i] == ['0']):
			outputfile = os.path.join(rootDir,patient_id[k],'notcompressed.%s.jpeg'%i)
		elif (category1[i] == ['1']) & (category2[i] == ['0']):
			outputfile = os.path.join(rootDir,patient_id[k],'pcompressed.%s.jpeg'%i)
		elif (category1[i] == ['0']) & (category2[i] == ['1']):
			outputfile = os.path.join(rootDir,patient_id[k],'pcompressed.%s.jpeg'%i)
		elif (category1[i] == ['1']) & (category2[i] == ['1']):
			outputfile = os.path.join(rootDir,patient_id[k],'compressed.%s.jpeg'%i)
		else:
			outputfile = os.path.join(rootDir,patient_id[k],'compressed.%s.jpeg'%i)

	#	print outputfile
		#plt.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, i]))
		plt.imshow(ArrayDicom[:, :, i])
		plt.savefig(outputfile)
	
	




