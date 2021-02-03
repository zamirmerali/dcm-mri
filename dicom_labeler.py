#Dicom Labeler
#Copyright, Zamir Merali, March 31, 2018
#2708, 80 John St. Toronto, ON, M5V3X4
#This module accomplishes the following task:
#Input: Many patient's ax_t2 images.
#The guts: Dicoms are normalized to slice thickness, a map is applied, more pre-processing. 
#Output: Each slice is displayed, awaits an input, and appends the class label to an array
#Output: Array with label for each slice

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


#Define list of patient ids
patient_id = []
#Define lists for filepaths
ax_t2_filepath = []
#Define the root directory
rootDir = '/Users/zamirmerali/aospine/sapien_tech/labeling/KAU_ax_forlabeling'
#Fill list of patient ids
patient_id = os.listdir(rootDir)
print patient_id
#Current patient id
ident = 22
print patient_id[ident]

#Function, take patient id, output list of file paths for ax t2 dicoms
def paths_ax_t2(patient_id):
	subDir = os.path.join(rootDir,patient_id,'proc_dicom','ax_t2')
	for root, dirs, files in os.walk(subDir):
		for fname in files:
			if fname.endswith('.dcm'):
				img_path = os.path.join(root,fname)
				ax_t2_filepath.append(img_path)
	return ax_t2_filepath

t2_filepath = paths_ax_t2(patient_id[ident])
print(len(t2_filepath))

#Get ref file
RefDs = pydicom.read_file(t2_filepath[0])
#print RefDs

#Load dimensions of the dicom sequence
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(t2_filepath))

#Load spacing values in mm
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

#Calculate axes for the array
x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# Make empty array based on extracted dimensions
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in t2_filepath:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, t2_filepath.index(filenameDCM)] = ds.pixel_array 

print(ConstPixelDims)
print(ConstPixelSpacing)

#generate plot 
#plt.figure(dpi=300)
#plt.axes().set_aspect('equal', 'datalim')
#plt.set_cmap(plt.gray())
#plt.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 5]))
#plt.show()
#Function, take numpy array with dicoms, output each slice in turn, ask for input, save input to the numpy array
plt.ion()
category = []
csvfile = os.path.join(rootDir,patient_id[ident],'compression_label.csv')
for i in range(len(t2_filepath)):
	#plt.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, i]))
	#plt.show()
	#plt.pause(0.001)
	plt.imshow(ArrayDicom[:, :, i])
	category.append(raw_input('compression?: '))
	print("Remaining: %d" % (len(t2_filepath)-i))


with open(csvfile, "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for val in category:
		writer.writerow([val])






