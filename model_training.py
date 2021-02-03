#Dependancies
import cv2
import numpy
import numpy as np
import os
import tensorflow as tf 
import sklearn.model_selection

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils
from random import shuffle
from tqdm import tqdm
#from matplotlib import pyplot as plt
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression

from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import History 
from keras.callbacks import CSVLogger

#Image input section
#define the folders where the training and testing images are located
train_dir = os.path.join('C:\projects','newtrainingdata')
test_dir = ''
#define the names of the classes, modify to suit the problem ie) improved vs. notimproved
class_1 = 'compressed' 
class_2 = 'pcompressed'
class_3 = 'notcompressed'

#define size that all iamges will be resized to
img_width = 299
img_height = 299 

#function: input patient_id, output list of ground truth labels
import csv
from numpy import genfromtxt

def get_labels(patient_id,location):
    csvpath = os.path.join(train_dir,location,patient_id,'compression_label.csv')
    csvlabels = genfromtxt(csvpath,delimiter=',')
    
    human_labels = []
    for x in csvlabels:
        if x == 0.0:
            human_labels.append(0)
        else:
            human_labels.append(1)
    
    return human_labels


#define a function that will read the class from the file name and output a classification vector
def label_img(img):
    word_label = img.split('.')[-3] #split filename by '.' and store class name
    #output a vector based on the class name
    if word_label == class_1:
        return 1
    elif word_label == class_2:
        return 1
    elif word_label == class_3:
        return 0 
    else:
        print('error: invalid class in filename')
        print(img)

 #define a function that creates an array of training images
def create_train_data():
    training_data = [] #create a blank array to be filled
    #loop to cycle through training images
    print(os.listdir(train_dir))
    for root, dirs, files in os.walk(train_dir):
        for fname in files:
            if fname.endswith('.jpeg') and fname.startswith('nolab') == False:
                #print(root)
                label = label_img(fname) #store the class vector
                path = os.path.join(train_dir,root,fname) #generate and store the full path for the image
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #read and store the image as grayscale
                #img = img[100:400,150:550]
                if not img is None:
                    img = cv2.resize(img, (img_width,img_height)) #resize the image
                    training_data.append([np.array(img), np.array(label)]) #add the image and the class vector to the data array
    np.save('train_data.npy', training_data) #save the data array
    return training_data #output the data array
#create the training data array
train_data = numpy.array(create_train_data())
#print(train_data) #output the data array
print(len(train_data))
print (train_data.shape)
# #Image processing section
# #Tasks: covert data to float32, normalize data...
train, test = train_test_split(train_data, train_size = 0.75, shuffle = True) 
print(len(train))
print(len(test))
print(train.shape)
print(test.shape)

train_x = np.array([i[0] for i in train]).reshape(-1,img_width,img_height,1)
train_y = np.array([i[1] for i in train])
test_x = np.array([i[0] for i in test]).reshape(-1,img_width,img_height,1)
test_y = np.array([i[1] for i in test])
print('Before dimensional expansion')
print(len(train_x))
print(train_x.shape)
print(len(train_y))
print(train_y.shape)
print(len(test_x))
print(test_x.shape)
print(len(test_y))
print(test_y.shape)
print('After dimensional expansion')
train_x_3 = np.repeat(train_x[:, :, :, :], 3, axis=3)
test_x_3 = np.repeat(test_x[:, :, :, :], 3, axis=3)
print(len(train_x_3))
print(train_x_3.shape)
print(len(test_x_3))
print(test_x_3.shape)
train_y = np_utils.to_categorical(train_y, 2)
test_y = np_utils.to_categorical(test_y, 2)
train_x_3 = train_x_3.astype('float32')
test_x_3 = test_x_3.astype('float32')
train_x_3 /= 255
test_x_3 /= 255


#Model Training Section
img_rows, img_cols, img_channel = 299,299,3
model_res50_conv = ResNet50(weights='imagenet', include_top=False)
model_res50_conv.summary()
#Change input format
input = Input(shape=(img_rows,img_cols,img_channel), name='image_input')
#Generate the convolutional model
output_res50_conv = model_res50_conv(input)
#Add fully-connected layers
x = GlobalAveragePooling2D(name='globalaveragepooling')(output_res50_conv)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu', name='fc2')(x)
x = Dropout(0.4)(x)
x = Dense(2, activation='sigmoid', name='predictions')(x)
INIT_LR = 1e-4
BS = 16
NUM_EPOCHS = 200
final_model = Model(input=input, output=x)
final_model.summary()
opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / config.NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
#fit the model
history = History()
csv_logger = CSVLogger('model5_1.log', separator=',', append=False)
train_datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=False)
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, write_graph=True, write_images=False)
tuner.search(train_ds,
             validation_data=test_ds,
             epochs=NUM_EPOCHS,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=p_factor)])
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BS,
	validation_data=valGen,
	validation_steps=totalVal // config.BS,
	epochs=config.NUM_EPOCHS)
#final_model.fit_generator(train_datagen.flow(train_x_3, train_y, batch_size=16), 
#    steps_per_epoch=len(train_x)/16, 
#    epochs=epochs, verbose=2, validation_data = (test_x_3, test_y), callbacks=[history, csv_logger])
#final_model.fit(train_x_3, train_y, batch_size=32, nb_epoch=10, verbose=1)

final_model.save('filename')
#print(final_model.history)