#Dependancies
import cv2
import numpy
import numpy as np
import os
import tensorflow as tf 
#import np_utils
import sklearn.model_selection
import pandas as pd

#import keras
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from scipy import interp
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils
from random import shuffle
from tqdm import tqdm
from matplotlib import pyplot as plt
from numpy import argmax
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
#from skimage import data, img_as_float
#from skimage import exposure
from itertools import groupby

from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
root_dir = R'C:\projects\newtestingdata'
#define size that all iamges will be resized to
img_width = 299
img_height = 299 
def load_the_model():
    model = load_model(R'filename')
    return model
def load_images(patient_id,location):
    patient_path = os.path.join(root_dir,location,patient_id)
    print(patient_path)
    test_data = [] #create blank array to be filled with images
    file_index = [] #blank array to be filled with the image number
    file_path = [] #blank array to be filled with file paths for the images
    for root, dirs, files in os.walk(patient_path):
        filename = []
        for fname in files: #terrible hacky code that sorts the files
            if fname.endswith('.jpeg') and fname.startswith('nolab') == True:
                filename.append(fname)
        for fname in sorted(filename, key = lambda x:int(x.split('.')[-2])):
            if fname.endswith('.jpeg') and fname.startswith('nolab') == True: #restrict to jpegs
                path = os.path.join(patient_path,fname) #create the path to the image
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #read and store the image as grayscale
                if not img is None:
                    img = cv2.resize(img, (img_width,img_height)) #resize the image
                    test_data.append([np.array(img)]) #add the image and the class vector to the data array
                    
    test_data = np.array([i[0] for i in test_data]).reshape(-1,img_width,img_height,1)
    test_data = np.repeat(test_data[:, :, :, :], 3, axis=3)
    test_data = test_data.astype('float32')
    test_data /= 255
    return test_data]
def gen_labels_for_patient(patient_images):
    prediction = []
    prediction = model.predict(patient_images)
    return prediction
#load the model
model = load_the_model()
 patient_id = 'TOH-002'
location = 'ax_foranalysis_combined'
patient_images = load_images(patient_id,location)
len(patient_images)
list_of_locations = os.listdir(root_dir)
print(list_of_locations)
list_of_patients = os.listdir(os.path.join(root_dir,list_of_locations[0]))
print(list_of_patients)
from numpy import genfromtxt

def get_labels(patient_id,location):
    csvpath = os.path.join(root_dir,location,patient_id,'compression_label.csv')
    csvlabels = genfromtxt(csvpath,delimiter=',')
    
    human_labels = []
    for x in csvlabels:
        if x == 0.0:
            human_labels.append(0)
        else:
            human_labels.append(1)
    
    return human_labels
patient_results = []

def longest_compression(compression_label):
    previous_level = 0
    segment_length = 1
    for i in range(len(compression_label)):
        curr_level = compression_label[i]
        if curr_level == 1 & previous_level == 1:
            segment_length = segment_length + 1
        previous_level = curr_level
    percent_segment_length = segment_length/len(compression_label)
    return percent_segment_length

tprs = [] #graph
base_fpr = np.linspace(0, 1, 101) #graph
plt.figure(figsize=(10,10)) #graph

for index, location in enumerate(list_of_locations):
    print(location)
    list_of_patients = os.listdir(os.path.join(root_dir,location))
    
    for i in range(0,len(list_of_patients)):
        patient_id = list_of_patients[i]
        count1_1 = 0 
        count1_0 = 0 
        count0_0 = 0 
        count0_1 = 0 

        patient_images = load_images(patient_id,location)
        computer_labels = gen_labels_for_patient(patient_images)
        prediction_class = np.argmax(computer_labels, axis=1)
        prediction_values = computer_labels[:,1]
        
        human_labels = get_labels(patient_id,location)

        print(prediction_class)
        print(human_labels)
        print(prediction_values)
        
        for j in range(0,len(prediction_class)):
            if human_labels[j] == 1 and prediction_class[j] == 1:
                count1_1 = count1_1 + 1
            elif human_labels[j] == 1 and prediction_class[j] == 0:
                count1_0 = count1_0 + 1
            elif human_labels[j] == 0 and prediction_class[j] == 0:
                count0_0 = count0_0 + 1
            elif human_labels[j] == 0 and prediction_class[j] == 1:
                count0_1 = count0_1 + 1
        
        fpr_keras, tpr_keras, _ = roc_curve(human_labels, prediction_values)
        auc_keras = auc(fpr_keras, tpr_keras)
        test_mean = np.mean(tpr_keras)
        plt.plot(fpr_keras, tpr_keras, 'b', alpha=0.05)
        tpr = interp(base_fpr, fpr_keras, tpr_keras)
        tpr[0] = 0.0
        tprs.append(tpr)
        print(tpr)
    
        print(count1_1,count1_0,count0_0,count0_1)
        print(len(prediction_class))
        print(auc_keras)
        
        patient_results.append([patient_id,count1_1,count1_0,count0_0,count0_1,len(prediction_class),auc_keras])     
tprs = np.array(tprs)
tprs = np.where(np.isnan(tprs), 1.0, tprs)
mean_tprs = tprs.mean(axis=0)
print(mean_tprs)
std = tprs.std(axis=0)
print(std)
q75, q25 = np.percentile(tprs, [75 ,25])
iqr = (q75 - q25)/2

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, 'b',color='green')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = 0.3)

plt.plot([0,1], [0,1],'r--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('testingcurve.png', dpi=300)
plt.show()
fig = plt.figure()
plt.savefig('testingcurve.png', dpi=fig.dpi)    
    